import math

from c2f import price
from c2f.price import Belief, Calibration, price_item, best_charge, accept_limit


CAL = Calibration(bias=1.0, sigma=0.4, p0=0.35, k=2.0)


def test_belief_median_and_quantiles():
    b = Belief.from_estimate({"t_low": 380, "t_mid": 420, "t_high": 450}, CAL)
    assert abs(b.median - 420) < 1e-9
    assert b.quantile(0.5) == b.median
    assert b.quantile(0.1) < b.median < b.quantile(0.9)


def test_belief_sigma_is_at_least_calibrated_and_widens_with_model_spread():
    narrow = Belief.from_estimate({"t_low": 419, "t_mid": 420, "t_high": 421}, CAL)
    wide = Belief.from_estimate({"t_low": 100, "t_mid": 420, "t_high": 1500}, CAL)
    assert narrow.sigma == CAL.sigma
    assert wide.sigma > CAL.sigma


def test_bias_scales_the_median():
    cal = Calibration(bias=0.8, sigma=0.4, p0=0.35, k=2.0)
    b = Belief.from_estimate({"t_low": 380, "t_mid": 500, "t_high": 600}, cal)
    assert abs(b.median - 400) < 1e-9


def test_beta_pulls_small_estimates_up_and_large_ones_down():
    sloped = Calibration(bias=1.0, sigma=0.4, beta=0.7, p0=0.35, k=2.0)
    small = Belief.from_estimate({"t_low": 10, "t_mid": 15, "t_high": 20}, sloped)
    large = Belief.from_estimate({"t_low": 700, "t_mid": 800, "t_high": 900}, sloped)
    assert small.median > 15  # below the pivot: pulled up
    assert large.median < 800  # above the pivot: pulled down
    at_pivot = Belief.from_estimate({"t_low": price.PIVOT_T, "t_mid": price.PIVOT_T, "t_high": price.PIVOT_T}, sloped)
    assert abs(at_pivot.median - price.PIVOT_T) < 1e-9  # bias keeps its meaning at the pivot


def test_beta_one_reproduces_the_pure_bias_model():
    est = {"t_low": 380, "t_mid": 500, "t_high": 600}
    old = Belief.from_estimate(est, Calibration(bias=1.3, sigma=0.4))
    new = Belief.from_estimate(est, Calibration(bias=1.3, sigma=0.4, beta=1.0))
    assert abs(old.median - new.median) < 1e-9 and old.sigma == new.sigma


def test_accept_limit_is_the_b_quantile_below_the_median():
    b = Belief(math.log(420), 0.4)
    assert abs(accept_limit(b) - b.quantile(price.B_QUANTILE)) < 1e-9
    assert accept_limit(b) < 420


def test_best_charge_positive_and_capped_at_a_max_q():
    # Step-revenue objective: R(a) = a * P(t >= a) + FRAUD_PAYOUT_FRAC * E[t; t < a]. Its
    # optimum may legitimately sit ABOVE the median (a fair charge is paid even when refused),
    # bounded by the A_MAX_Q plateau cap.
    from c2f.price import A_MAX_Q
    b = Belief(math.log(1000), 0.4)
    a = best_charge(b, CAL)
    assert 300 < a <= b.quantile(A_MAX_Q) + 1e-9


def test_best_charge_ignores_the_fitted_acceptance_curve():
    # p0/k are OUT of the objective: the fitted p0 * r^-k curve was a censoring artifact
    # (k < 1 claimed revenue rises with the overcharge ratio forever) - measured payout above
    # t is a flat FRAUD_PAYOUT_FRAC * t regardless of r, so the charge must not move with p0/k.
    b = Belief(math.log(1000), 0.4)
    stingy = Calibration(bias=1.0, sigma=0.4, p0=0.05, k=2.0)
    generous = Calibration(bias=1.0, sigma=0.4, p0=0.6, k=0.5)
    assert abs(best_charge(b, stingy) - best_charge(b, generous)) < 1e-9


def test_risk_aversion_knob_still_shades_the_charge():
    # Production runs pure EV (RISK_AVERSION = 0, portfolio argument in price.py), but the
    # knob must keep working for autotune sweeps: an explicit risk_aversion shades the charge.
    b = Belief(math.log(1000), 0.6)
    assert best_charge(b, CAL, risk_aversion=1.0) < best_charge(b, CAL, risk_aversion=0.0)


def test_covered_item_a_below_mid_b_below_mid():
    est = {"covered": True, "related": True, "t_low": 380, "t_mid": 420, "t_high": 450}
    a, b = price_item(est, CAL)
    # a may sit above t_mid now (paid-even-when-refused makes that correct); it stays under
    # the A_MAX_Q cap of the belief. b = Q(1/3) is always below the median.
    from c2f.price import A_MAX_Q
    assert 0 < a <= Belief.from_estimate(est, CAL).quantile(A_MAX_Q) + 1e-9
    assert 0 < b < 420


def test_uncovered_item_b_zero_a_small():
    est = {"covered": False, "related": True, "t_low": 0, "t_mid": 0, "t_high": 0, "t_if_covered": 400}
    a, b = price_item(est, CAL)
    assert b == 0
    assert 0 < a < 400


def test_uncovered_without_guess_charges_zero():
    est = {"covered": False, "related": False, "t_low": 0, "t_mid": 0, "t_high": 0}
    assert price_item(est, CAL) == (0.0, 0.0)


def test_never_negative_or_non_finite():
    est = {"covered": True, "related": True, "t_low": 500, "t_mid": 100, "t_high": 50}
    a, b = price_item(est, CAL)
    assert math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0


def test_calibration_is_read_from_file_clamped_or_default(tmp_path, monkeypatch):
    monkeypatch.setattr(price, "CALIBRATION_PATH", tmp_path / "missing.json")
    assert price.calibration() == price.DEFAULT_CALIBRATION
    (tmp_path / "c.json").write_text('{"bias": 9.0, "sigma": 0.5, "p0": 0.2, "k": 1.0}')
    monkeypatch.setattr(price, "CALIBRATION_PATH", tmp_path / "c.json")
    c = price.calibration()
    assert c.bias == price.BIAS_RANGE[1] and c.sigma == 0.5 and c.p0 == 0.2 and c.k == 1.0
    (tmp_path / "c.json").write_text("garbage")
    assert price.calibration() == price.DEFAULT_CALIBRATION


def test_best_charge_never_exceeds_the_plateau_cap():
    # The old rail was "never above the median", needed because the k < 1 acceptance fit
    # rewarded charging the moon. The step objective has an interior optimum; the remaining
    # guard is the A_MAX_Q cap on how hard we lean on the lenient-field plateau.
    from c2f.price import A_MAX_Q
    b = Belief(math.log(1000), 1.0)
    cal = Calibration(bias=1.0, sigma=0.4, p0=0.35, k=0.5)
    assert best_charge(b, cal) <= b.quantile(A_MAX_Q) + 1e-9
    assert best_charge(b, cal, risk_aversion=0.0) <= b.quantile(A_MAX_Q) + 1e-9


def test_bundled_replacement_without_component_coverage_evidence_uncovered():
    """Bundled line with unclear component coverage must not be priced as covered.

    Scenario: Flat-rate replacement of boiler, flue, storage tank and pipework
    adjustment, where coverage is NOT established for every component.
    Expected: Entire line treated as uncovered with a=0, b=0.
    """
    # Model says covered, but reason doesn't establish coverage for all components
    est = {
        "index": 1,
        "covered": True,
        "related": True,
        "_description": "Flat-rate replacement of boiler, flue, storage tank and pipework adjustment",
        "clause": "Section 3",
        "reason": "Replacement of heating system",
        "t_low": 3000,
        "t_mid": 3500,
        "t_high": 4000,
        "t_if_covered": 3500,  # What it would cost if covered
        "cap_uncertain": False,
    }
    a, b = price_item(est, CAL)
    # Since bundle coverage validation fails, enforce a=0, b=0
    assert b == 0.0, "Bundle with unclear component coverage must have b=0"
    assert a == 0.0, "Bundle with unclear component coverage must have a=0 (strict gate)"


def test_bundled_replacement_with_explicit_component_coverage_treated_as_covered():
    """Bundled line where all components are explicitly covered is priced normally.

    Scenario: Flat-rate functional unit replacement where evidence establishes
    that the complete unit required replacement and every bundled operation is covered.
    Expected: Line treated as covered, normal (a, b) pricing applies.
    """
    est = {
        "index": 1,
        "covered": True,
        "related": True,
        "_description": "Complete boiler replacement including flue, tank and pipework adjustment",
        "clause": "Section 3: Amount of indemnity",
        "reason": "Boiler, flue, tank and pipework all covered as functional heating unit",
        "t_low": 3000,
        "t_mid": 3500,
        "t_high": 4000,
        "t_if_covered": 0,
        "cap_uncertain": False,
    }
    a, b = price_item(est, CAL)
    # Explicit component coverage should allow normal pricing
    assert b > 0, "Bundle with explicit component coverage should have b > 0"
    assert a > 0 and a < 5000, "Covered bundle should price based on belief"


def test_bundled_with_separate_component_prices():
    """Bundled line with separate component prices and partial coverage.

    When an invoice shows separate prices for components of a bundle,
    but coverage is not established for all components, the bundle fails validation.
    Expected: a=0, b=0 (strict gate on incomplete component coverage).
    """
    # Single bundled line with separate prices but incomplete coverage
    est = {
        "index": 1,
        "covered": True,
        "related": True,
        "_description": "Boiler (€2000) + flue work (€500) + pipework (€600)",
        "clause": "Section 3",
        "reason": "Boiler and pipework covered; flue work not established",
        "t_low": 2000,
        "t_mid": 2500,
        "t_high": 2800,
        "t_if_covered": 0,
    }
    a, b = price_item(est, CAL)
    # Incomplete component coverage (flue work "not established") means bundle uncovered
    assert b == 0.0, "Bundle with partial coverage should have b=0"
    assert a == 0.0, "Bundle with partial coverage should have a=0"
