"""Regression test for the Game 6 t-estimation failure.

Item 1 ("surge-damaged home electronics ... quantity 2"): the invoice/description gave no
brand, age, TV size, speaker spec or stated value. The full pass invented a 65" TV + 5.1
speaker system and priced t_mid ~= 1500, t_high 1700 - the actual priced row that went out
was charge_price=1375.00, acceptance_limit=1850.69 (runs/game_06.json, priced_full[0]).
Public results imply the real fair value was below EUR 900.

Item 2 ("diagnostic surge-failure report and technician call-out, quantity 3"): the full
pass called it covered (t_mid=170), the fast pass called it not covered
(t_if_covered=120) - a straight disagreement that the old pipeline ignored because the
fast pass was never compared against the full one. Public results imply t < EUR 45,
probably zero.

This test locks in the fix: the LLM prompt tells the model not to invent specifics, and
c2f.price.price_all takes the fast pass as a disagreement check on the full pass so a
coverage split forces a conservative price without needing another model call.
"""

import json

import pytest

from c2f import price as price_mod
from c2f.llm import SYSTEM
from c2f.price import Calibration, price_all
from c2f.submit import ROOT


@pytest.fixture(autouse=True)
def _fixed_calibration(monkeypatch):
    """Pin the calibration: the numbers below are about the disagreement check, not about
    whatever bias/sigma the latest game refit (runs/calibration.json drifts every round)."""
    monkeypatch.setattr(price_mod, "calibration", lambda: Calibration(bias=1.0, sigma=0.4, p0=0.35, k=2.0))

GAME_06_LOG = ROOT / "runs" / "game_06.json"

# The full pass's actual (buggy) Game 6 estimate for both items.
FULL_ITEMS = [
    {
        "index": 1,
        "covered": True,
        "related": True,
        "t_low": 1200,
        "t_mid": 1500,
        "t_high": 1700,
        "t_if_covered": 0,
        "reason": 'Mid-range 65" TV ~EUR 800 + 5.1 speakers ~EUR 700 incl. VAT',
    },
    {
        "index": 2,
        "covered": True,
        "related": True,
        "t_low": 120,
        "t_mid": 170,
        "t_high": 220,
        "t_if_covered": 300,
        "reason": "One necessary diagnostics/call-out; three billed excessive",
    },
]

# The fast pass's actual Game 6 estimate for the same case (runs/game_06.json, model_fast).
FAST_ITEMS = [
    {
        "index": 1,
        "covered": True,
        "related": True,
        "t_low": 300,
        "t_mid": 700,
        "t_high": 1100,
        "t_if_covered": 0,
        "reason": "appliances damaged by overvoltage; market mid-range replacement",
    },
    {
        "index": 2,
        "covered": False,
        "related": True,
        "t_low": 0,
        "t_mid": 0,
        "t_high": 0,
        "t_if_covered": 120,
        "reason": "technician diagnostic/call-out is a non-indemnifiable service",
    },
]


def test_prompt_forbids_inventing_specs_and_flags_cap_uncertainty():
    assert "NEVER INVENT SPECIFICS" in SYSTEM
    assert "cheapest reasonable standard replacement" in SYSTEM.lower() or "CHEAPEST reasonable standard" in SYSTEM
    assert "cap_uncertain" in SYSTEM
    assert "conservative" in SYSTEM.lower()


def test_diagnostic_item_disagreement_forces_uncovered_pricing():
    """Item 2: full says covered, fast says not - the split alone must zero out b."""
    rows = price_all(FULL_ITEMS, other_output={"items": FAST_ITEMS})
    item2 = next(r for r in rows if r["index"] == 2)
    assert item2["acceptance_limit"] == 0.0
    # a rejected fraud costs nothing, so charging conservatively is fine - never paying out is not.


def test_electronics_item_bounds_drop_substantially_below_the_game_06_output():
    """Item 1: the fast pass's much lower, spec-free estimate must pull the priced row down
    well below the actual (buggy) Game 6 submission of a=1375.00, b=1850.69."""
    rows = price_all(FULL_ITEMS, other_output={"items": FAST_ITEMS})
    item1 = next(r for r in rows if r["index"] == 1)
    assert item1["charge_price"] < 1000
    assert item1["acceptance_limit"] < 900  # public results imply the real t was below EUR 900


def test_without_a_second_opinion_pricing_is_unchanged_not_a_regression():
    """No fast estimate available (e.g. it failed): price_all must behave exactly as before -
    the disagreement check is additive, not a new requirement on every call."""
    rows_alone = price_all(FULL_ITEMS)
    rows_with_identical_other = price_all(FULL_ITEMS, other_output={"items": FULL_ITEMS})
    assert rows_alone == rows_with_identical_other


def test_game_06_log_reproduces_the_reported_bug_before_the_fix():
    """Sanity check that the fixture above matches what actually went out in Game 6. That run
    predates this fix (it used the old three-vote ensemble, since removed - see git history);
    the FULL_ITEMS/FAST_ITEMS fixtures above reproduce its estimates against today's pipeline."""
    log = json.loads(GAME_06_LOG.read_text())
    priced_full = log["priced_full"]
    assert priced_full[0]["charge_price"] == 1375.0
    assert priced_full[0]["acceptance_limit"] == 1850.69
    assert log["ensemble"]["items"][0]["t_mid"] == 1500
    assert "65" in log["ensemble"]["items"][0]["reason"]
    assert log["model_fast"]["output"]["items"][1]["covered"] is False
