"""The decision layer.

These tests encode the three results in `docs/DESIGN.md` as assertions, because
each of them contradicts an intuition someone will eventually try to "fix".
"""

from __future__ import annotations

import pytest

from app.c2f.decision.optimizer import choose_a, choose_b, decide, expected_revenue
from app.c2f.models import ACCEPT_THRESHOLD, Calibration, ItemInference, LineItem, PriceQuantiles
from app.c2f.probability.survival import flat_distribution, quantiles_from_median

LADDER = (280.0, 340.0, 430.0, 540.0, 680.0)


def item(quantity: float = 1.0) -> LineItem:
    return LineItem(item_id="windshield", description="Replacement windshield", quantity=quantity)


def inference(p_valid: float = 0.95, multiplier: float = 1.0) -> ItemInference:
    return ItemInference(
        item_id="windshield",
        p_valid=p_valid,
        unit_quantiles=PriceQuantiles.from_values(LADDER),
        skeptic_multiplier=multiplier,
    )


# ---------- R3: p_valid decides b and leaves a alone ----------


def test_p_valid_does_not_move_the_optimal_charge():
    """`S(a) = q * S_+(a)`, so q is a constant factor and cancels out of the argmax.

    This is the result most likely to be "corrected" by someone who assumes a
    doubtful item should be charged less. It should not be.
    """
    charges = [
        choose_a(flat_distribution(q, median=1000.0, sigma_log=0.5))
        for q in (1.0, 0.8, 0.5, 0.2, 0.05)
    ]
    for value in charges[1:]:
        assert value == pytest.approx(charges[0])


def test_b_is_the_33rd_percentile_when_validity_is_certain():
    dist = flat_distribution(1.0, median=1000.0, sigma_log=0.5)
    b = choose_b(dist)
    assert dist.cdf_positive(b) == pytest.approx(1.0 / 3.0, abs=1e-6)


@pytest.mark.parametrize(
    "p_valid,expected_level",
    [(1.0, 1 / 3), (0.9, 1 - (2 / 3) / 0.9), (0.8, 1 - (2 / 3) / 0.8), (0.7, 1 - (2 / 3) / 0.7)],
)
def test_b_tracks_the_two_thirds_bar(p_valid, expected_level):
    dist = flat_distribution(p_valid, median=1000.0, sigma_log=0.5)
    b = choose_b(dist)
    assert dist.cdf_positive(b) == pytest.approx(expected_level, abs=1e-6)
    # The bar itself: S(b) must sit right at 2/3.
    assert dist.survival(b) == pytest.approx(ACCEPT_THRESHOLD, abs=1e-6)


@pytest.mark.parametrize("p_valid", [0.666, 0.5, 0.3, 0.0])
def test_b_is_exactly_zero_below_the_bar(p_valid):
    """No positive price can clear 2/3 once q does. Paying the lawyer is cheaper."""
    assert choose_b(flat_distribution(p_valid, median=1000.0)) == 0.0


def test_b_collapses_as_p_valid_approaches_the_bar():
    """No cliff, but a steep one.

    Just above 2/3 the required confidence level is ~1e-9, so `b` comes off the
    far lower tail - a few percent of the median rather than literally zero. It
    still satisfies `S(b) >= 2/3`, so paying it is correct; it is simply a tiny
    exposure. Below the bar it is exactly zero.
    """
    limits = [
        choose_b(flat_distribution(q, median=1000.0))
        for q in (1.0, 0.9, 0.8, 0.7, ACCEPT_THRESHOLD + 1e-9)
    ]
    assert limits == sorted(limits, reverse=True)
    assert limits[-1] < 0.05 * 1000.0
    assert choose_b(flat_distribution(ACCEPT_THRESHOLD - 1e-9, median=1000.0)) == 0.0


# ---------- R2: uncertainty moves a up and b down ----------


@pytest.mark.parametrize(
    "sigma_log,expected_percentile",
    [(0.3, 0.17), (0.4, 0.24), (0.6, 0.37), (0.8, 0.50), (1.0, 0.62), (1.2, 0.72)],
)
def test_optimal_charge_percentile_rises_with_uncertainty(sigma_log, expected_percentile):
    """`argmax a*S(a)` solves `hazard(z) = sigma` for the standard normal.

    The quantile ladder is only a 5-knot approximation of a lognormal, so this
    checks the shape rather than the exact analytic root.
    """
    dist = flat_distribution(1.0, median=1000.0, sigma_log=sigma_log)
    percentile = dist.cdf_positive(choose_a(dist))
    assert percentile == pytest.approx(expected_percentile, abs=0.07)


def test_a_and_b_diverge_as_uncertainty_grows():
    tight = flat_distribution(1.0, median=1000.0, sigma_log=0.3)
    wide = flat_distribution(1.0, median=1000.0, sigma_log=1.2)

    # Tight belief: charge below the acceptance limit. Wide belief: above it.
    assert choose_a(tight) < choose_b(tight)
    assert choose_a(wide) > choose_b(wide)


def test_b_below_a_is_not_repaired():
    """`b < a` is a legitimate output. Nothing in the pipeline may normalise it."""
    decision = decide(
        item(),
        ItemInference(
            item_id="windshield",
            p_valid=1.0,
            unit_quantiles=quantiles_from_median(1000.0, sigma_log=1.2),
        ),
    )
    assert decision.b < decision.a
    assert any("not a bug" in note for note in decision.notes)


# ---------- R1: the fair-zone term needs no opponent model ----------


def test_expected_revenue_is_never_negative():
    """As issuer the worst case is collecting nothing - there is no penalty."""
    dist = flat_distribution(0.4, median=500.0, sigma_log=0.7)
    for price in (0.01, 10.0, 500.0, 2000.0, 100_000.0):
        assert expected_revenue(price, dist) >= 0.0


def test_an_acceptance_model_only_ever_raises_the_charge():
    """G(a) adds a non-negative fraud-zone term, so it cannot lower the optimum."""
    dist = flat_distribution(0.95, median=1000.0, sigma_log=0.5)
    baseline = choose_a(dist)
    generous = choose_a(dist, acceptance=lambda _price: 0.9)
    assert generous >= baseline


def test_soft_opponents_push_the_charge_well_past_the_median():
    dist = flat_distribution(0.95, median=1000.0, sigma_log=0.4)
    assert choose_a(dist, acceptance=lambda _price: 1.0) > dist.median


def test_a_hard_opponent_model_leaves_the_honest_optimum_alone():
    dist = flat_distribution(0.95, median=1000.0, sigma_log=0.5)
    assert choose_a(dist, acceptance=lambda _price: 0.0) == pytest.approx(choose_a(dist))


# ---------- bounds, caps, knobs ----------


def test_charge_never_exceeds_four_times_the_median():
    """`c >= 4t`, so anything above that is waste even in the best case."""
    dist = flat_distribution(0.95, median=1000.0, sigma_log=1.5)
    assert choose_a(dist, acceptance=lambda _price: 1.0) <= dist.median * 4.0 + 1e-6


def test_cap_clamps_the_search():
    dist = flat_distribution(0.95, median=1000.0, sigma_log=1.5)
    assert choose_a(dist, acceptance=lambda _price: 1.0, cap=1200.0) <= 1200.0 + 1e-6


def test_worthless_item_still_gets_a_charge():
    """A rejected charge costs the issuer nothing, so a=0 is a free option thrown away."""
    dist = flat_distribution(0.0, median=800.0, sigma_log=0.5)
    assert choose_a(dist) > 0.0


def test_lambda_knobs_scale_the_outputs():
    dist = flat_distribution(1.0, median=1000.0, sigma_log=0.5)
    assert choose_a(dist, lambda_a=1.2) == pytest.approx(choose_a(dist) * 1.2)
    assert choose_b(dist, lambda_b=0.8) == pytest.approx(choose_b(dist) * 0.8)


def test_lambda_b_cannot_resurrect_a_zero_limit():
    """The 2/3 constant comes from the payoff table and is not ours to tune."""
    dist = flat_distribution(0.4, median=1000.0)
    assert choose_b(dist, lambda_b=5.0) == 0.0


# ---------- the assembled decision ----------


def test_decide_reports_its_own_inputs():
    decision = decide(item(quantity=3.0), inference())
    assert decision.item_id == "windshield"
    assert decision.quantity == 3.0
    assert decision.q50_gross == pytest.approx(LADDER[2] * 3.0)
    assert decision.s_at_a == pytest.approx(
        flat_distribution(0.95, decision.q50_gross, decision.sigma_log).survival(decision.a),
        abs=0.02,
    )
    assert "median" in decision.reason


def test_decide_scales_with_quantity():
    one = decide(item(quantity=1.0), inference())
    four = decide(item(quantity=4.0), inference())
    assert four.a == pytest.approx(one.a * 4.0)
    assert four.b == pytest.approx(one.b * 4.0)


def test_decide_notes_a_zero_limit():
    decision = decide(item(), inference(p_valid=0.4))
    assert decision.b == 0.0
    assert any("2/3" in note for note in decision.notes)


def test_decide_applies_calibration():
    plain = decide(item(), inference())
    shifted = decide(item(), inference(), calibration=Calibration(mu_shift=0.8))
    assert shifted.a == pytest.approx(plain.a * 0.8, rel=0.02)
