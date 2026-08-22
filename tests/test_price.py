from c2f.price import price_item, tri_quantile


def test_tri_quantile_endpoints_and_mode():
    assert tri_quantile(100, 200, 300, 0.0) == 100
    assert tri_quantile(100, 200, 300, 1.0) == 300
    # mode of a symmetric triangle is the median
    assert abs(tri_quantile(100, 200, 300, 0.5) - 200) < 1e-9


def test_tri_quantile_third_below_mode():
    q = tri_quantile(380, 420, 450, 1 / 3)
    assert 380 < q < 420


def test_degenerate_range():
    assert tri_quantile(420, 420, 420, 1 / 3) == 420


def test_covered_item_a_below_mid_b_is_third_quantile():
    est = {"covered": True, "related": True, "t_low": 380, "t_mid": 420, "t_high": 450}
    a, b = price_item(est, scale=1.0)
    assert 380 <= a < 420
    assert 380 < b < 420
    assert b == round(tri_quantile(380, 420, 450, 1 / 3), 2)


def test_uncovered_item_b_zero_a_small():
    est = {"covered": False, "related": True, "t_low": 0, "t_mid": 0, "t_high": 0, "t_if_covered": 400}
    a, b = price_item(est)
    assert b == 0
    assert 0 < a < 400


def test_uncovered_without_guess_charges_zero():
    est = {"covered": False, "related": False, "t_low": 0, "t_mid": 0, "t_high": 0}
    assert price_item(est) == (0.0, 0.0)


def test_tight_range_charges_nearly_mid():
    est = {"covered": True, "related": True, "t_low": 419, "t_mid": 420, "t_high": 421}
    a, _ = price_item(est)
    assert a >= 419


def test_never_negative_or_non_finite():
    est = {"covered": True, "related": True, "t_low": 500, "t_mid": 100, "t_high": 50}
    a, b = price_item(est)
    assert a >= 0 and b >= 0


def test_b_scale_pushes_accept_limit_up_only():
    est = {"covered": True, "related": True, "t_low": 380, "t_mid": 420, "t_high": 450}
    a1, b1 = price_item(est, scale=1.0)
    a2, b2 = price_item(est, scale=1.3)
    assert a1 == a2  # the charge is not scaled
    assert abs(b2 - 1.3 * b1) < 0.02


def test_default_b_scale_is_read_from_calibration_or_default(tmp_path, monkeypatch):
    from c2f import price

    monkeypatch.setattr(price, "CALIBRATION_PATH", tmp_path / "missing.json")
    assert price.b_scale() == price.B_SCALE_DEFAULT
    (tmp_path / "c.json").write_text('{"b_scale": 9.0}')
    monkeypatch.setattr(price, "CALIBRATION_PATH", tmp_path / "c.json")
    assert price.b_scale() == price.B_SCALE_RANGE[1]  # clamped
    (tmp_path / "c.json").write_text("garbage")
    assert price.b_scale() == price.B_SCALE_DEFAULT
