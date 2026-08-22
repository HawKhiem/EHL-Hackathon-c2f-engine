"""The pre-submit check. Its whole job is to stop a factor-of-ten going out."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.c2f.decision.guardrails import blocking, check, repair
from app.c2f.models import Calibration, ItemDecision, ItemInference, LineItem
from app.c2f.probability.survival import quantiles_from_median


def good(item_id: str = "1", **overrides) -> ItemDecision:
    base = ItemDecision(
        item_id=item_id,
        a=400.0,
        b=350.0,
        s_at_a=0.7,
        s_at_b=0.67,
        sigma_log=0.4,
        q50_gross=430.0,
        p_valid=0.95,
        quantity=1.0,
    )
    return replace(base, **overrides) if overrides else base


ITEMS = [LineItem(item_id="1", description="windshield")]


def test_a_clean_submission_passes():
    assert check(ITEMS, [good()]) == []


def test_b_below_a_is_never_flagged():
    """The one invariant we deliberately do not enforce. See docs/DESIGN.md R2."""
    assert check(ITEMS, [good(a=900.0, b=200.0)]) == []
    assert repair([good(a=900.0, b=200.0)])[0].b == 200.0


def test_missing_decision_blocks():
    violations = check([*ITEMS, LineItem(item_id="2", description="labour")], [good()])
    assert any(v.rule == "item_coverage" for v in blocking(violations))


def test_unknown_item_blocks():
    violations = check(ITEMS, [good(), good(item_id="ghost")])
    assert any("unknown" in v.detail for v in blocking(violations))


def test_duplicate_item_id_blocks():
    violations = check(ITEMS, [good(), good()])
    assert any("duplicate" in v.detail for v in blocking(violations))


def test_non_finite_blocks():
    violations = check(ITEMS, [good(a=float("nan"))])
    assert any(v.rule == "finite" for v in blocking(violations))


def test_charge_above_four_times_median_blocks():
    violations = check(ITEMS, [good(a=430.0 * 4.5)])
    assert any(v.rule == "charge_bounds" for v in blocking(violations))


def source(quantity_free_unit_q50: float = 430.0, **overrides) -> ItemInference:
    base = ItemInference(
        item_id="1",
        p_valid=0.95,
        unit_quantiles=quantiles_from_median(quantity_free_unit_q50, sigma_log=0.4),
    )
    return replace(base, **overrides) if overrides else base


@pytest.mark.parametrize("quantity", [2.0, 5.0, 20.0])
def test_skipped_quantity_multiply_blocks_at_any_quantity(quantity):
    """The failure this whole module exists for: a unit price sent as a gross total.

    Caught by the derivation, so it trips at quantity 2 as reliably as at 20 -
    a magnitude heuristic could only ever catch the large ones.
    """
    item = LineItem(item_id="1", description="labour hours", quantity=quantity)
    skipped = good(a=430.0, q50_gross=430.0)  # forgot to multiply by quantity
    violations = check([item], [skipped], inferences=[source()])
    assert any(v.rule == "gross_total" for v in blocking(violations))


def test_doubled_quantity_multiply_blocks():
    item = LineItem(item_id="1", description="labour hours", quantity=3.0)
    doubled = good(a=1000.0, q50_gross=430.0 * 9.0)  # quantity applied twice
    violations = check([item], [doubled], inferences=[source()])
    assert any(v.rule == "gross_total" for v in blocking(violations))


def test_correct_derivation_passes():
    item = LineItem(item_id="1", description="labour hours", quantity=3.0)
    correct = good(a=1000.0, b=900.0, q50_gross=430.0 * 3.0)
    assert check([item], [correct], inferences=[source()]) == []


def test_derivation_check_accounts_for_calibration_and_skeptic():
    item = LineItem(item_id="1", description="labour hours", quantity=2.0)
    expected = 430.0 * 2.0 * 0.7 * 0.9
    decision = good(a=400.0, b=350.0, q50_gross=expected)
    violations = check(
        [item],
        [decision],
        inferences=[source(skeptic_multiplier=0.7)],
        calibration=Calibration(mu_shift=0.9),
    )
    assert violations == []


def test_positive_limit_below_the_two_thirds_bar_blocks():
    violations = check(ITEMS, [good(p_valid=0.4, b=100.0)])
    assert any(v.rule == "accept_threshold" for v in blocking(violations))


def test_zero_charge_warns_but_does_not_block():
    violations = check(ITEMS, [good(a=0.0)])
    assert any(v.rule == "zero_charge" for v in violations)
    assert blocking(violations) == []


def test_invoice_total_mismatch_warns():
    violations = check(ITEMS, [good(a=400.0)], invoice_total=40_000.0)
    assert any(v.rule == "invoice_total" for v in violations)
    assert blocking(violations) == []


def test_repair_fixes_what_it_can():
    broken = [
        good(item_id="1", a=float("inf"), b=-5.0),
        good(item_id="2", p_valid=0.4, b=100.0),
        good(item_id="3", a=0.0),
    ]
    fixed = {d.item_id: d for d in repair(broken)}

    # A non-finite charge falls back to the median, not to the 4x ceiling:
    # garbage input should not become our most aggressive charge.
    assert fixed["1"].a == 430.0
    assert fixed["1"].b == 0.0
    assert fixed["2"].b == 0.0
    assert fixed["3"].a == 430.0
    assert all(any("repaired" in n for n in fixed[i].notes) for i in ("2", "3"))


def test_repair_output_passes_check():
    items = [LineItem(item_id=str(i), description="x") for i in ("1", "2", "3")]
    broken = [
        good(item_id="1", a=float("nan"), b=float("nan")),
        good(item_id="2", p_valid=0.3, b=900.0),
        good(item_id="3", a=430.0 * 9.0),
    ]
    assert blocking(check(items, repair(broken))) == []


def test_repair_leaves_clean_decisions_untouched():
    clean = good()
    assert repair([clean])[0] is clean
