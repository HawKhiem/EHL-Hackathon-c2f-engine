"""The probability layer. A silent factor-of-ten lives or dies here."""

from __future__ import annotations

import math

import pytest

from app.c2f.models import QUANTILE_LEVELS, Calibration, ItemInference, PriceQuantiles
from app.c2f.probability.survival import (
    FairValueDistribution,
    flat_distribution,
    quantiles_from_median,
)

LADDER = (280.0, 340.0, 430.0, 540.0, 680.0)


def make_dist(p_valid: float = 1.0, ladder: tuple[float, ...] = LADDER) -> FairValueDistribution:
    return FairValueDistribution(p_valid, ladder)


def test_cdf_is_exact_at_the_knots():
    dist = make_dist()
    for level, price in zip(QUANTILE_LEVELS, LADDER, strict=True):
        assert dist.cdf_positive(price) == pytest.approx(level, abs=1e-9)


def test_tails_are_continuous_with_the_body():
    """Each tail is fitted to its own adjacent pair, so no jump at q10 or q90."""
    dist = make_dist()
    for price in (LADDER[0], LADDER[-1]):
        below = dist.cdf_positive(price * 0.9999)
        above = dist.cdf_positive(price * 1.0001)
        assert abs(above - below) < 1e-3


def test_survival_is_monotone_and_bracketed():
    dist = make_dist(p_valid=0.9)
    assert dist.survival(0.0) == 1.0
    assert dist.survival(-5.0) == 1.0
    assert dist.survival(1e9) == pytest.approx(0.0, abs=1e-6)

    previous = 1.0
    for price in (1.0, 100.0, 280.0, 430.0, 680.0, 5000.0):
        current = dist.survival(price)
        assert current <= previous + 1e-12
        assert 0.0 <= current <= 0.9 + 1e-12
        previous = current


def test_quantile_inverts_the_cdf():
    dist = make_dist()
    for level in (0.02, 0.1, 0.2, 0.33, 0.5, 0.75, 0.9, 0.98):
        price = dist.quantile_positive(level)
        assert dist.cdf_positive(price) == pytest.approx(level, abs=1e-6)


def test_quantile_clamps_outside_the_unit_interval():
    dist = make_dist()
    assert dist.quantile_positive(0.0) == 0.0
    assert dist.quantile_positive(-1.0) == 0.0
    assert dist.quantile_positive(1.0) > LADDER[-1]


@pytest.mark.parametrize(
    "raw",
    [
        (0.0, 0.0, 0.0, 0.0, 0.0),  # all zero
        (100.0, 100.0, 100.0, 100.0, 100.0),  # every knot tied
        (500.0, 400.0, 430.0, 300.0, 900.0),  # inverted
        (float("nan"), 340.0, 430.0, float("inf"), 680.0),  # non-finite
    ],
)
def test_degenerate_ladders_still_produce_a_usable_curve(raw):
    """Model output does all of these. None of them may raise."""
    dist = FairValueDistribution(1.0, raw)
    assert dist.median > 0
    assert math.isfinite(dist.sigma_log)
    previous = 1.0
    for price in (1.0, 50.0, 500.0, 5000.0):
        current = dist.survival(price)
        assert 0.0 <= current <= previous + 1e-12
        previous = current


def test_quantity_multiplies_the_ladder_in_code():
    """The gross total is quantity x unit, computed here and never in a prompt."""
    inference = ItemInference(
        item_id="1", p_valid=1.0, unit_quantiles=PriceQuantiles.from_values(LADDER)
    )
    unit = FairValueDistribution.from_inference(inference, quantity=1.0)
    gross = FairValueDistribution.from_inference(inference, quantity=4.0)

    assert gross.median == pytest.approx(unit.median * 4.0)
    # Scaling prices cannot change the shape of the belief.
    assert gross.sigma_log == pytest.approx(unit.sigma_log)


@pytest.mark.parametrize("quantity", [0.0, -3.0, float("nan")])
def test_bad_quantity_falls_back_to_one(quantity):
    inference = ItemInference(
        item_id="1", p_valid=1.0, unit_quantiles=PriceQuantiles.from_values(LADDER)
    )
    dist = FairValueDistribution.from_inference(inference, quantity=quantity)
    assert dist.median == pytest.approx(LADDER[2])


def _with_skeptic(multiplier: float) -> FairValueDistribution:
    return FairValueDistribution.from_inference(
        ItemInference(
            item_id="1",
            p_valid=1.0,
            unit_quantiles=PriceQuantiles.from_values(LADDER),
            skeptic_multiplier=multiplier,
        )
    )


def test_skeptic_multiplier_shrinks_and_is_clamped():
    assert _with_skeptic(0.7).median == pytest.approx(LADDER[2] * 0.7)
    # A multiplier above 1 would let the skeptic argue prices *up*. Clamped.
    assert _with_skeptic(3.0).median == pytest.approx(LADDER[2])
    # And it cannot shrink a price to nothing.
    assert _with_skeptic(0.0).median == pytest.approx(LADDER[2] * 0.5)


def test_calibration_shifts_the_median_and_stretches_the_spread():
    inference = ItemInference(
        item_id="1", p_valid=1.0, unit_quantiles=PriceQuantiles.from_values(LADDER)
    )
    base = FairValueDistribution.from_inference(inference)
    shifted = FairValueDistribution.from_inference(inference, calibration=Calibration(mu_shift=0.8))
    widened = FairValueDistribution.from_inference(
        inference, calibration=Calibration(sigma_scale=2.0)
    )

    assert shifted.median == pytest.approx(base.median * 0.8)
    assert shifted.sigma_log == pytest.approx(base.sigma_log)

    assert widened.median == pytest.approx(base.median)
    assert widened.sigma_log == pytest.approx(base.sigma_log * 2.0)


def test_flat_distribution_matches_its_requested_shape():
    dist = flat_distribution(0.9, median=1000.0, sigma_log=0.5)
    assert dist.median == pytest.approx(1000.0)
    assert dist.sigma_log == pytest.approx(0.5, abs=1e-9)
    assert dist.quantile_positive(0.5) == pytest.approx(1000.0)


def test_quantiles_from_median_is_ordered():
    ladder = quantiles_from_median(500.0, 0.4).values
    assert list(ladder) == sorted(ladder)
    assert ladder[2] == pytest.approx(500.0)


def test_price_quantiles_from_mapping_is_tolerant():
    assert PriceQuantiles.from_mapping(
        {"q10": 1, "q25": 2, "q50": 3, "q75": 4, "q90": 5}
    ).values == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert PriceQuantiles.from_mapping(
        {"0.1": 1, "0.25": 2, "0.5": 3, "0.75": 4, "0.9": 5}
    ).values == (1.0, 2.0, 3.0, 4.0, 5.0)
    with pytest.raises(ValueError):
        PriceQuantiles.from_mapping({"q10": 1})


def test_sample_threshold_respects_the_hurdle():
    dist = make_dist(p_valid=0.6)
    assert dist.sample_threshold(0.9, 0.5) == 0.0  # drew "not valid"
    assert dist.sample_threshold(0.1, 0.5) == pytest.approx(LADDER[2])
