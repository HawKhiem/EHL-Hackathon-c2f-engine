"""The semantic layer. Every test here is a way a real reply goes wrong."""

from __future__ import annotations

import asyncio
import json

import pytest

from app.c2f.inference.analyse import CaseBundle, analyse_case, heuristic_inferences
from app.c2f.inference.schemas import (
    DEFAULT_P_VALID,
    extract_json,
    index_by_item,
    merge_inferences,
)
from app.c2f.models import LineItem
from app.llm.base import Completion

ITEMS = [
    LineItem(item_id="1", description="Windschutzscheibe", quantity=1.0),
    LineItem(item_id="2", description="Arbeitszeit Monteur", quantity=3.5, unit="h"),
]


def validity_payload() -> list[dict]:
    return [
        {"item_id": "1", "p_valid": 0.96, "p_covered": 0.98, "p_related": 0.97, "evidence": "e"},
        {"item_id": "2", "p_valid": 0.9, "evidence": "labour follows the glass"},
    ]


def pricing_payload() -> list[dict]:
    return [
        {"item_id": "1", "q10": 280, "q25": 340, "q50": 430, "q75": 540, "q90": 680},
        {"item_id": "2", "q10": 60, "q25": 70, "q50": 85, "q75": 100, "q90": 130},
    ]


class ScriptedProvider:
    """Returns a canned reply per system prompt, so calls stay distinguishable."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, replies: dict[str, str | Exception], *, delay: float = 0.0) -> None:
        self.replies = replies
        self.delay = delay
        self.seen: list[str] = []

    async def complete(self, messages, *, system=None, max_tokens=16_000) -> Completion:
        which = self._which(system or "")
        self.seen.append(which)
        if self.delay:
            await asyncio.sleep(self.delay)
        reply = self.replies.get(which, "")
        if isinstance(reply, Exception):
            raise reply
        if reply == "__refused__":
            return Completion(content="declined", model=self.model, refused=True)
        return Completion(content=reply, model=self.model)

    async def stream(self, messages, *, system=None, max_tokens=64_000):  # pragma: no cover
        yield ""
        raise NotImplementedError

    @staticmethod
    def _which(system: str) -> str:
        if "claims examiner" in system:
            return "validity"
        if "repair-cost assessor" in system:
            return "pricing"
        return "skeptic"


def bundle() -> CaseBundle:
    return CaseBundle(
        case_id="0",
        items=list(ITEMS),
        policy="Comprehensive glass cover applies.",
        description="Stone chip cracked the windshield on the A9.",
    )


# ---------- extract_json ----------


@pytest.mark.parametrize(
    "text",
    [
        '[{"item_id": "1", "p_valid": 0.9}]',
        'Sure!\n```json\n[{"item_id": "1", "p_valid": 0.9}]\n```\nHope that helps.',
        'Here you go: [{"item_id": "1", "p_valid": 0.9}] - let me know.',
        '{"items": [{"item_id": "1", "p_valid": 0.9}]}',
    ],
)
def test_extract_json_survives_prose_and_fences(text):
    payload = extract_json(text)
    assert payload is not None
    assert index_by_item(payload)["1"]["p_valid"] == 0.9


def test_extract_json_is_string_aware():
    """A brace inside an evidence string must not unbalance the scan."""
    text = '[{"item_id": "1", "evidence": "clause {3} says \\"covered\\"", "p_valid": 0.8}]'
    assert index_by_item(extract_json(text))["1"]["p_valid"] == 0.8


@pytest.mark.parametrize("text", ["", "no json here", "{unclosed", "[1, 2,"])
def test_extract_json_returns_none_rather_than_raising(text):
    assert extract_json(text) is None


def test_extract_json_skips_a_broken_object_and_finds_the_next():
    assert extract_json('{"bad": } {"item_id": "1"}') == {"item_id": "1"}


# ---------- index_by_item ----------


def test_index_by_item_accepts_the_three_shapes_models_return():
    rows = [{"item_id": "1", "p_valid": 0.9}]
    as_list = index_by_item(rows)
    as_wrapped = index_by_item({"items": rows})
    as_keyed = index_by_item({"1": {"p_valid": 0.9}})
    assert as_list["1"]["p_valid"] == as_wrapped["1"]["p_valid"] == as_keyed["1"]["p_valid"]


def test_index_by_item_falls_back_to_position_when_ids_are_dropped():
    assert set(index_by_item([{"p_valid": 0.9}, {"p_valid": 0.8}])) == {"1", "2"}


def test_index_by_item_ignores_junk():
    assert index_by_item("nope") == {}
    assert index_by_item([1, 2, 3]) == {}


# ---------- merge ----------


def test_merge_produces_one_inference_per_item_in_order():
    merged = merge_inferences(
        ITEMS, index_by_item(validity_payload()), index_by_item(pricing_payload())
    )
    assert [i.item_id for i in merged] == ["1", "2"]
    assert merged[0].p_valid == 0.96
    assert merged[0].unit_quantiles.q50 == 430.0
    assert not any(i.degraded for i in merged)


def test_merge_covers_an_item_the_model_dropped():
    """A missing item invalidates the whole submission, so it always gets a row."""
    partial = index_by_item([pricing_payload()[0]])
    merged = merge_inferences(ITEMS, index_by_item(validity_payload()), partial)
    assert len(merged) == 2
    assert merged[1].degraded
    assert merged[1].p_valid == 0.9  # validity survived even though pricing did not


def test_merge_defaults_p_valid_above_the_two_thirds_bar():
    """A blanket b=0 would pay the 1.5x penalty on every covered item."""
    merged = merge_inferences(ITEMS, {}, index_by_item(pricing_payload()))
    assert all(i.p_valid == DEFAULT_P_VALID for i in merged)
    assert DEFAULT_P_VALID > 2 / 3


def test_merge_accepts_percentages():
    rows = index_by_item([{"item_id": "1", "p_valid": 96}, {"item_id": "2", "p_valid": 90}])
    merged = merge_inferences(ITEMS, rows, index_by_item(pricing_payload()))
    assert merged[0].p_valid == pytest.approx(0.96)


def test_merge_repairs_a_partial_quantile_ladder():
    rows = index_by_item([{"item_id": "1", "q50": 400}, {"item_id": "2", "q50": 90}])
    merged = merge_inferences(ITEMS, index_by_item(validity_payload()), rows)
    assert merged[0].unit_quantiles.q50 == pytest.approx(400.0)
    assert merged[0].degraded  # we invented the spread, so say so
    assert merged[0].unit_quantiles.q10 < merged[0].unit_quantiles.q90


def test_merge_clamps_a_skeptic_that_argues_prices_up():
    skeptic = index_by_item([{"item_id": "1", "suggested_multiplier": 2.5}])
    merged = merge_inferences(
        ITEMS, index_by_item(validity_payload()), index_by_item(pricing_payload()), skeptic
    )
    # Stored raw; the clamp to [0.5, 1.0] is applied when the distribution is built.
    assert merged[0].skeptic_multiplier == 2.5


# ---------- analyse_case ----------


@pytest.mark.asyncio
async def test_analyse_case_runs_every_call_and_merges():
    provider = ScriptedProvider(
        {
            "validity": json.dumps(validity_payload()),
            "pricing": json.dumps(pricing_payload()),
            "skeptic": json.dumps([{"item_id": "1", "suggested_multiplier": 0.9}]),
        }
    )
    result = await analyse_case(bundle(), provider=provider)

    assert sorted(provider.seen) == ["pricing", "skeptic", "validity"]
    assert result.calls_ok == {"validity": True, "pricing": True, "skeptic": True}
    assert [i.item_id for i in result.inferences] == ["1", "2"]
    assert result.inferences[0].skeptic_multiplier == 0.9


@pytest.mark.asyncio
async def test_one_dead_call_does_not_lose_the_round():
    provider = ScriptedProvider(
        {
            "validity": RuntimeError("provider exploded"),
            "pricing": json.dumps(pricing_payload()),
            "skeptic": "not json at all",
        }
    )
    result = await analyse_case(bundle(), provider=provider)

    assert result.calls_ok["validity"] is False
    assert result.calls_ok["skeptic"] is False
    assert len(result.inferences) == 2
    assert all(i.p_valid == DEFAULT_P_VALID for i in result.inferences)
    assert result.inferences[0].unit_quantiles.q50 == 430.0  # pricing still landed


@pytest.mark.asyncio
async def test_a_refusal_is_treated_as_a_missing_call():
    """Opus 5 can decline with HTTP 200 and stop_reason=refusal."""
    provider = ScriptedProvider(
        {"validity": "__refused__", "pricing": json.dumps(pricing_payload()), "skeptic": ""}
    )
    result = await analyse_case(bundle(), provider=provider)
    assert result.calls_ok["validity"] is False
    assert len(result.inferences) == 2


@pytest.mark.asyncio
async def test_the_deadline_is_enforced_not_hoped_for():
    provider = ScriptedProvider(
        {"validity": json.dumps(validity_payload()), "pricing": "", "skeptic": ""},
        delay=0.5,
    )
    result = await analyse_case(bundle(), provider=provider, timeout=0.05)
    assert not any(result.calls_ok.values())
    assert len(result.inferences) == 2  # still submittable
    assert all(i.degraded for i in result.inferences)


@pytest.mark.asyncio
async def test_empty_invoice_is_not_an_error():
    result = await analyse_case(CaseBundle(case_id="0", items=[]), provider=ScriptedProvider({}))
    assert result.inferences == []


@pytest.mark.asyncio
async def test_skeptic_can_be_switched_off():
    provider = ScriptedProvider(
        {"validity": json.dumps(validity_payload()), "pricing": json.dumps(pricing_payload())}
    )
    await analyse_case(bundle(), provider=provider, with_skeptic=False)
    assert "skeptic" not in provider.seen


def test_heuristic_inferences_are_submittable_with_no_model_at_all():
    inferences = heuristic_inferences(ITEMS)
    assert [i.item_id for i in inferences] == ["1", "2"]
    assert all(i.degraded for i in inferences)
    assert all(i.unit_quantiles.q50 > 0 for i in inferences)
