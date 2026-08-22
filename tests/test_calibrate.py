import math

from c2f.calibrate import fit_bias_sigma, fit_acceptance


def test_fit_recovers_a_downward_bias_from_two_sided_bounds():
    # true t = 0.8 * mid with little noise; bounds bracket t tightly
    rows = []
    for mid in (100, 200, 500, 1000, 1500, 80, 300, 700):
        t = 0.8 * mid
        rows.append({"t_lo": t * 0.97, "t_hi": t * 1.03, "t_mid": mid})
    bias, sigma = fit_bias_sigma(rows)
    assert abs(bias - 0.8) < 0.05
    assert sigma <= 0.2


def test_fit_uses_upper_bounds_alone():
    # every label says t < 0.6 * mid: the fit must move the bias down, not stay at 1
    rows = [{"t_lo": 0.3 * m, "t_hi": 0.6 * m, "t_mid": m} for m in (100, 250, 400, 900, 1200)]
    bias, _ = fit_bias_sigma(rows)
    assert bias < 0.7


def test_fit_acceptance_curve_decays():
    pts = [(1.1, 0.30), (1.5, 0.16), (2.0, 0.09), (4.0, 0.02), (1.2, 0.25), (3.0, 0.04)]
    p0, k = fit_acceptance(pts)
    assert 0.2 < p0 < 0.5 and 1.0 < k < 3.0
    assert p0 * 1.1 ** -k > p0 * 3.0 ** -k
