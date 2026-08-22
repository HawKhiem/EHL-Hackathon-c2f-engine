"""Run the semantic layer: concurrent calls, hard deadline, per-item degradation.

Nothing here may raise. A round with one bad LLM reply must still submit; a round
with three bad replies must still submit. The only thing that ends a round is the
clock, and the clock is enforced here rather than hoped for.

ASYNC109 is suppressed on purpose. Ruff wants the caller to wrap these in
`asyncio.timeout` instead of passing a deadline in, but an outside timeout
cancels the whole coroutine and throws away the replies that *did* arrive.
Per-call deadlines inside are the only version that still submits.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.c2f.inference.prompts import (
    PRICING_SYSTEM,
    SKEPTIC_SYSTEM,
    VALIDITY_SYSTEM,
    case_context,
)
from app.c2f.inference.schemas import extract_json, index_by_item, merge_inferences
from app.c2f.models import ItemInference, LineItem
from app.llm import LLMProvider, get_llm

log = logging.getLogger(__name__)

#: Whole-case budget for the semantic layer. The 60s round has to fit key fetch,
#: decrypt, parse, this, the optimiser and two POSTs, so this is the slack we can
#: afford, not the time the model would like.
DEFAULT_TIMEOUT: float = 25.0
#: The skeptic is the first thing cut when time is short: it only shrinks prices,
#: so losing it is a smaller error than losing validity or pricing.
SKEPTIC_TIMEOUT: float = 18.0


@dataclass(slots=True)
class CaseBundle:
    """A decrypted case, parsed. No file handles - parsing already happened."""

    case_id: str
    items: list[LineItem]
    policy: str = ""
    description: str = ""
    image_paths: list[str] = field(default_factory=list)
    stated_total: float | None = None


@dataclass(slots=True)
class AnalysisResult:
    inferences: list[ItemInference]
    #: Which calls came back. Goes straight into the round log - when a round
    #: scores badly we need to know whether the model was wrong or absent.
    calls_ok: dict[str, bool]
    elapsed: float


async def _call(
    provider: LLMProvider,
    name: str,
    system: str,
    user: str,
    *,
    timeout: float,  # noqa: ASYNC109 - see module docstring
    max_tokens: int,
) -> tuple[str, dict[str, Mapping[str, Any]]]:
    """One structured call. Returns an empty mapping on any failure at all."""
    try:
        async with asyncio.timeout(timeout):
            completion = await provider.complete(
                [{"role": "user", "content": user}],
                system=system,
                max_tokens=max_tokens,
            )
    except TimeoutError:
        log.warning("c2f.%s timed out after %.1fs", name, timeout)
        return name, {}
    except Exception:  # noqa: BLE001 - a provider error must not end the round
        log.exception("c2f.%s failed", name)
        return name, {}

    if completion.refused:
        # Opus 5 can decline with HTTP 200 and stop_reason="refusal".
        log.warning("c2f.%s declined: %s", name, completion.content[:200])
        return name, {}

    payload = extract_json(completion.content)
    if payload is None:
        log.warning("c2f.%s returned no parsable JSON (%d chars)", name, len(completion.content))
        return name, {}
    return name, index_by_item(payload)


async def analyse_case(
    bundle: CaseBundle,
    *,
    provider: LLMProvider | None = None,
    timeout: float = DEFAULT_TIMEOUT,  # noqa: ASYNC109 - see module docstring
    with_skeptic: bool = True,
) -> AnalysisResult:
    """Every call at once, then merge whatever arrived.

    Concurrent because they are independent: validity reads the policy, pricing
    reads the market, the skeptic reads the invoice against itself. Chaining them
    would triple the latency and buy nothing.
    """
    provider = provider or get_llm()
    loop = asyncio.get_running_loop()
    started = loop.time()

    if not bundle.items:
        return AnalysisResult([], {}, 0.0)

    validity_user = case_context(bundle.items, policy=bundle.policy, description=bundle.description)
    pricing_user = case_context(bundle.items, description=bundle.description, include_policy=False)

    tasks = [
        _call(
            provider,
            "validity",
            VALIDITY_SYSTEM,
            validity_user,
            timeout=timeout,
            max_tokens=8_000,
        ),
        _call(
            provider,
            "pricing",
            PRICING_SYSTEM,
            pricing_user,
            timeout=timeout,
            max_tokens=8_000,
        ),
    ]
    if with_skeptic:
        tasks.append(
            _call(
                provider,
                "skeptic",
                SKEPTIC_SYSTEM,
                validity_user,
                timeout=min(SKEPTIC_TIMEOUT, timeout),
                max_tokens=6_000,
            )
        )

    settled = await asyncio.gather(*tasks, return_exceptions=True)

    rows: dict[str, dict[str, Mapping[str, Any]]] = {}
    for outcome in settled:
        if isinstance(outcome, BaseException):
            log.exception("c2f analysis task crashed", exc_info=outcome)
            continue
        name, mapping = outcome
        rows[name] = mapping

    inferences = merge_inferences(
        bundle.items,
        rows.get("validity", {}),
        rows.get("pricing", {}),
        rows.get("skeptic", {}),
    )

    return AnalysisResult(
        inferences=inferences,
        calls_ok={name: bool(mapping) for name, mapping in rows.items()},
        elapsed=loop.time() - started,
    )


def heuristic_inferences(items: Sequence[LineItem]) -> list[ItemInference]:
    """What the T+8 safety-net submission runs on: no model output at all.

    Wide and mildly confident, which yields a high charge and a low acceptance
    limit - the cautious direction on both sides of the game.
    """
    return merge_inferences(items, {}, {}, {})
