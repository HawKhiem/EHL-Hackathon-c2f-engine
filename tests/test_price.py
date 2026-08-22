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


def test_accept_limit_is_the_b_quantile_below_the_median():
    b = Belief(math.log(420), 0.4)
    assert abs(accept_limit(b) - b.quantile(price.B_QUANTILE)) < 1e-9
    assert accept_limit(b) < 420


def test_best_charge_sits_below_the_median_and_above_zero():
    b = Belief(math.log(1000), 0.4)
    a = best_charge(b, CAL)
    assert 300 < a < 1000


def test_best_charge_drops_when_the_market_accepts_less_fraud():
    b = Belief(math.log(1000), 0.4)
    stingy = Calibration(bias=1.0, sigma=0.4, p0=0.05, k=2.0)
    generous = Calibration(bias=1.0, sigma=0.4, p0=0.6, k=0.5)
    assert best_charge(b, stingy) < best_charge(b, generous)


def test_best_charge_drops_when_less_sure():
    sure = Belief(math.log(1000), 0.2)
    unsure = Belief(math.log(1000), 0.8)
    assert best_charge(unsure, CAL) < best_charge(sure, CAL)


def test_covered_item_a_below_mid_b_below_mid():
    est = {"covered": True, "related": True, "t_low": 380, "t_mid": 420, "t_high": 450}
    a, b = price_item(est, CAL)
    assert 0 < a < 420
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


def test_best_charge_never_exceeds_median_even_with_slow_acceptance_decay_and_wide_belief():
    b = Belief(math.log(1000), 1.0)
    cal = Calibration(bias=1.0, sigma=0.4, p0=0.35, k=0.5)
    assert best_charge(b, cal) <= 1000 + 1e-9
    assert best_charge(b, cal, risk_aversion=0.0) <= 1000 + 1e-9
