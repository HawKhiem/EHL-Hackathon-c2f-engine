"""Regression test for the Game 10 covered-but-zero valuation bug.

Item 3 ("Compensation for stolen watch"): the full pass never landed, and the only
submission was the fast pass's answer - covered=True, related=True, but
t_low=t_mid=t_high=0 (runs/game_10.json, model_fast.output.items[2]). The old pipeline
only logged a warning for this shape (c2f.llm._validate_items) and c2f.price.price_item
then treated a covered item with mid<=0 exactly like an uncovered one: a=0, b=0. Public
results proved the watch's fair value was >= EUR 7,225 (an "expensive watch" per the
description), so the b=0 refused every fair charge and the a=0 also missed the charge
opportunity - runs/postmortem_game_10.json records this as EUR 61,527.29 of wrongful-
rejection penalty plus EUR 115,600.00 of missed-charge cost on this one item.

This test locks in the fix:
- c2f.validate.validate_items flags an all-zero covered+related item as invalid, not just
  a warning (point 1 of the fix).
- c2f.price.price_item refuses to price that state at all - it raises InvalidEstimateError
  instead of silently producing a=0, b=0 (point 2).
- c2f.run.resolve_invalid_items repairs an invalid item using the other model lane's valid
  estimate, or a positive t_if_covered, before pricing ever sees it (point 4's fallback
  order, minus the extra repair LLM call this team does not run - OpenAI-only, no added
  model round trip).
- a valid full-pass result replaces an invalid fast one, and vice versa never happens
  (point 5).
"""

from __future__ import annotations

import json

import pytest

from c2f.price import InvalidEstimateError, price_item
from c2f.run import merge_estimates, resolve_invalid_items
from c2f.submit import ROOT
from c2f.validate import invalid_indices, validate_items

GAME_10_LOG = ROOT / "runs" / "game_10.json"

# The fast pass's actual (buggy) Game 10 item 3 answer - the stolen watch.
WATCH_INVALID = {
    "index": 3,
    "covered": True,
    "related": True,
    "clause": "4.2.2 / 7.1.1 / 7.1.9",
    "t_low": 0,
    "t_mid": 0,
    "t_high": 0,
    "t_if_covered": 0,
    "reason": "value depends on valuables sub-limit (see note)",
}

CASE = {
    "description": (
        "An expensive watch was taken in a robbery while the policyholder was on holiday "
        "abroad. They were threatened with a weapon and the robbers took the watch off "
        "their wrist along with some cash they were carrying."
    ),
    "items": [
        {"index": 3, "description": "Compensation for stolen watch (premium replacement model, upgrade on the one taken)", "quantity": 1, "unit": "pcs"},
    ],
}


def test_game_10_log_reproduces_the_reported_bug_before_the_fix():
    """Sanity check that the fixture above matches what actually went out in Game 10."""
    if not GAME_10_LOG.exists():
        import pytest

        pytest.skip("run log not present: runs/ is claim data and is not checked in")
    log = json.loads(GAME_10_LOG.read_text())
    item3 = next(it for it in log["model_fast"]["output"]["items"] if it["index"] == 3)
    assert item3["covered"] is True and item3["related"] is True
    assert item3["t_low"] == item3["t_mid"] == item3["t_high"] == 0
    priced3 = next(r for r in log["priced_fast"] if r["index"] == 3)
    assert priced3 == {"index": 3, "charge_price": 0.0, "acceptance_limit": 0.0}


def test_covered_and_related_all_zero_t_is_invalid():
    errors = validate_items({"items": [WATCH_INVALID]})
    assert invalid_indices(errors) == {3}


def test_uncovered_all_zero_t_remains_valid():
    uncovered = {"index": 5, "covered": False, "related": False, "t_low": 0, "t_mid": 0, "t_high": 0, "t_if_covered": 0}
    assert validate_items({"items": [uncovered]}) == []


def test_price_item_refuses_to_silently_price_the_invalid_state():
    with pytest.raises(InvalidEstimateError):
        price_item(WATCH_INVALID)


def test_only_the_invalid_item_is_sent_for_repair():
    """resolve_invalid_items must leave already-valid items untouched and only replace the
    broken index - it must not touch index 4 just because index 3 is invalid."""
    valid_other = {"index": 4, "covered": True, "related": True, "t_low": 350, "t_mid": 400, "t_high": 450}
    items = [WATCH_INVALID, valid_other]
    resolved, fatal = resolve_invalid_items(items, other_by_idx={})
    item4 = next(it for it in resolved if it["index"] == 4)
    assert item4 == valid_other


def test_repaired_output_is_merged_at_the_correct_index():
    other_repair = {"index": 3, "covered": True, "related": True, "t_low": 6000, "t_mid": 7225, "t_high": 9000}
    resolved, fatal = resolve_invalid_items([WATCH_INVALID], other_by_idx={3: other_repair})
    assert fatal == []
    item3 = next(it for it in resolved if it["index"] == 3)
    assert item3["t_mid"] == 7225


def test_repaired_watch_produces_positive_a_and_b():
    other_repair = {"index": 3, "covered": True, "related": True, "t_low": 6000, "t_mid": 7225, "t_high": 9000}
    resolved, _ = resolve_invalid_items([WATCH_INVALID], other_by_idx={3: other_repair})
    a, b = price_item(resolved[0])
    assert a > 0 and b > 0


def test_falls_back_to_positive_t_if_covered_when_no_other_lane():
    watch_with_guess = {**WATCH_INVALID, "t_if_covered": 7225}
    resolved, fatal = resolve_invalid_items([watch_with_guess], other_by_idx={})
    assert fatal == []
    item3 = resolved[0]
    assert item3["t_low"] == item3["t_mid"] == item3["t_high"] == 7225
    a, b = price_item(item3)
    assert a > 0 and b > 0


def test_no_valid_valuation_anywhere_is_reported_fatal_not_disguised_as_uncovered():
    resolved, fatal = resolve_invalid_items([WATCH_INVALID], other_by_idx={})
    assert fatal and fatal[0]["index"] == 3
    # still priceable (never a=0,b=0 disguised as a normal decision - it's flagged fatal)
    a, b = price_item(resolved[0])
    assert a >= 0 and b >= 0


def test_valid_full_result_replaces_invalid_fast_result():
    fast_out = {"items": [WATCH_INVALID]}
    full_out = {"items": [{"index": 3, "covered": True, "related": True, "t_low": 6000, "t_mid": 7225, "t_high": 9000}]}
    fast_by_idx = {it["index"]: it for it in fast_out["items"]}
    resolved, fatal = resolve_invalid_items(merge_estimates(CASE, full_out), other_by_idx=fast_by_idx)
    assert fatal == []
    assert resolved[0]["t_mid"] == 7225


def test_invalid_fast_result_cannot_replace_a_valid_full_result():
    full_valid = {"index": 3, "covered": True, "related": True, "t_low": 6000, "t_mid": 7225, "t_high": 9000}
    fast_by_idx = {3: WATCH_INVALID}
    resolved, fatal = resolve_invalid_items([full_valid], other_by_idx=fast_by_idx)
    assert fatal == []
    assert resolved[0] == full_valid


def test_final_submission_contains_every_expected_index_exactly_once():
    case = {**CASE, "items": [{"index": i, "description": "x", "quantity": 1, "unit": "pcs"} for i in (1, 2, 3, 4)]}
    full_out = {
        "items": [
            {"index": 1, "covered": False, "related": False, "t_low": 0, "t_mid": 0, "t_high": 0},
            {"index": 2, "covered": True, "related": True, "t_low": 10, "t_mid": 20, "t_high": 30},
            WATCH_INVALID,
            # index 4 missing entirely from the model output
        ]
    }
    merged = merge_estimates(case, full_out)
    resolved, fatal = resolve_invalid_items(merged, other_by_idx={})
    assert sorted(it["index"] for it in resolved) == [1, 2, 3, 4]
