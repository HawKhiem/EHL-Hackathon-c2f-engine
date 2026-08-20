"""Anthropic Messages API provider — the default.

Notes that matter for Claude Opus 5 and are easy to get wrong:

* Adaptive thinking (`thinking={"type": "adaptive"}`) is the current API.
  `budget_tokens` is removed on this model family and returns a 400.
* Safety classifiers can *decline* a request. That is an HTTP 200 with
  `stop_reason == "refusal"`, not an exception — so we check `stop_reason`
  before reading `content`.
* `fallbacks="default"` re-runs a declined request on another model
  server-side. It needs a beta flag that may not be enabled on every key,
  so the first rejection downgrades the client for the process lifetime
  instead of failing the request.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic

from app.llm.base import Completion, Message

log = logging.getLogger(__name__)

# Scalar "default" fallback mode — routes by refusal category server-side.
# Distinct from the array form, which uses the -2026-06-01 header.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, *, api_key: str, model: str, enable_refusal_fallback: bool = True) -> None:
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to .env "
                "(get one from the sponsor desk or console.anthropic.com)."
            )
        self.model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._fallback_enabled = enable_refusal_fallback

    # ---------------------------------------------------------------- kwargs

    def _request_kwargs(
        self, messages: Sequence[Message], system: str | None, max_tokens: int
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": list(messages),
            # Adaptive thinking: the model decides how much to think.
            # "summarized" so a UI can show reasoning; the default is omitted.
            "thinking": {"type": "adaptive", "display": "summarized"},
        }
        if system:
            kwargs["system"] = system
        if self._fallback_enabled:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def _disable_fallback(self, err: Exception) -> bool:
        """If the beta was rejected, drop it and let the caller retry once."""
        if not self._fallback_enabled:
            return False
        blob = str(err).lower()
        if "fallback" in blob or "beta" in blob:
            log.warning(
                "Server-side refusal fallback rejected by the API (%s). "
                "Disabling it for this process; set LLM_ENABLE_REFUSAL_FALLBACK=false "
                "to silence this.",
                type(err).__name__,
            )
            self._fallback_enabled = False
            return True
        return False

    # ---------------------------------------------------------------- calls

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 16_000,
    ) -> Completion:
        for attempt in (1, 2):
            kwargs = self._request_kwargs(messages, system, max_tokens)
            try:
                response = await self._client.beta.messages.create(**kwargs)
                break
            except anthropic.BadRequestError as err:
                if attempt == 1 and self._disable_fallback(err):
                    continue
                raise
        else:  # pragma: no cover - loop always breaks or raises
            raise RuntimeError("unreachable")

        # Check stop_reason BEFORE reading content — a refusal can carry an
        # empty content list, so indexing content[0] would blow up here.
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            return Completion(
                content=(
                    "The model declined this request"
                    + (f" (category: {category})" if category else "")
                    + ". Rephrasing usually helps."
                ),
                model=response.model,
                refused=True,
            )

        text = "".join(block.text for block in response.content if block.type == "text")
        return Completion(content=text, model=response.model)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 64_000,
    ) -> AsyncIterator[str]:
        for attempt in (1, 2):
            kwargs = self._request_kwargs(messages, system, max_tokens)
            try:
                async with self._client.beta.messages.stream(**kwargs) as stream:
                    async for text in stream.text_stream:
                        yield text

                    # A mid-stream decline shows up on the final message.
                    final = await stream.get_final_message()
                    if final.stop_reason == "refusal":
                        yield "\n\n⚠ The model declined to continue this request."
                return
            except anthropic.BadRequestError as err:
                # Only safe to retry if nothing was emitted yet, which is the
                # case for a request-validation error on the beta flag.
                if attempt == 1 and self._disable_fallback(err):
                    continue
                raise
