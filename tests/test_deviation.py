import math

from c2f import deviation as D


def test_distance_is_zero_inside_the_bracket():
    # the market proved 100 <= t < 200; anything in there is consistent, not an error
    assert D.censored(100.0, 100.0, 200.0) == 0.0
    assert D.censored(150.0, 100.0, 200.0) == 0.0
    assert D.censored(199.9, 100.0, 200.0) == 0.0


def test_distance_is_signed_log_gap_to_the_bound_that_broke():
    assert D.censored(50.0, 100.0, 200.0) == -math.log(2)   # under
    assert D.censored(400.0, 100.0, 200.0) == math.log(2)   # over
    assert D.censored(200.0, 100.0, 200.0) == math.log(1.0)  # t_hi is exclusive: over by 0


def test_open_bracket_can_only_prove_under():
    assert D.censored(50.0, 100.0, None) < 0
    assert D.censored(10_000.0, 100.0, None) == 0.0  # nothing proves an upper bound


def test_abstention_has_no_log_distance():
    assert D.censored(0.0, 100.0, 200.0) is None
    assert D.censored(-1.0, 100.0, 200.0) is None


def test_t_star_is_the_geometric_midpoint_and_falls_back_to_the_floor():
    assert D.t_star(100.0, 400.0) == 200.0
    assert D.t_star(100.0, None) == 100.0
    assert D.t_star(0.0, None) is None


def test_regret_charges_half_for_a_refused_fair_charge():
    # t >= 100, a charge of 80 is proven fair, our limit of 50 would refuse it
    r = D.regret(a=0.0, b=50.0, lo=100.0, hi=None, charges=[80.0])
    assert r["n_refused_fair"] == 1
    assert r["b_refused_fair"] == 40.0


def test_regret_charges_the_full_amount_for_accepted_fraud():
    # t < 100, a charge of 300 is proven fraud, our limit of 500 would pay it
    r = D.regret(a=0.0, b=500.0, lo=0.0, hi=100.0, charges=[300.0])
    assert r["n_took_fraud"] == 1
    assert r["b_took_fraud"] == 300.0


def test_accepted_fraud_is_capped_at_four_t():
    r = D.regret(a=0.0, b=10_000.0, lo=0.0, hi=100.0, charges=[9_000.0])
    assert r["b_took_fraud"] == 400.0  # CAP_MULT * t_hi


def test_a_charge_inside_the_bracket_costs_nothing_either_way():
    r = D.regret(a=150.0, b=150.0, lo=100.0, hi=200.0, charges=[150.0])
    assert r["b_refused_fair"] == 0.0 and r["b_took_fraud"] == 0.0 and r["a_forgone"] == 0.0


def test_a_below_the_proven_floor_is_money_left_on_every_opponents_table():
    r = D.regret(a=80.0, b=0.0, lo=100.0, hi=None, charges=[])
    assert r["a_forgone"] == 20.0 * D.N_OPPONENTS


def test_rows_score_t_a_and_b_against_the_same_bracket():
    truth = {7: {"1": {"t_lo": 100.0, "t_hi": 200.0, "charges": [90.0, 300.0]}}}
    est = {7: {1: {"index": 1, "covered": True, "related": True,
                   "t_low": 40, "t_mid": 50, "t_high": 60, "_description": "Drying fan"}}}
    board = {7: {1: {"index": 1, "charge_price": 50.0, "acceptance_limit": 400.0}}}
    (r,) = D.rows(truth, est, board)
    assert r["kind"] == "priced" and r["bucket"] == "drying/remediation"
    assert r["dev_t"] == -math.log(2) and r["dev_a"] == -math.log(2)  # both under t_lo
    assert r["dev_b"] == math.log(2)                                  # b over t_hi
    assert r["regret"]["n_refused_fair"] == 0                         # b=400 pays the fair 90
    assert r["regret"]["n_took_fraud"] == 1                           # ...and the fraudulent 300
    assert r["regret"]["a_forgone"] == 50.0 * D.N_OPPONENTS


def test_an_item_we_called_uncovered_is_a_coverage_miss_not_a_price_error():
    truth = {7: {"1": {"t_lo": 100.0, "t_hi": None, "charges": []}}}
    est = {7: {1: {"index": 1, "covered": False, "related": True, "t_mid": 0,
                   "t_if_covered": 90, "_description": "Service technician hours"}}}
    board = {7: {1: {"index": 1, "charge_price": 81.0, "acceptance_limit": 0.0}}}
    (r,) = D.rows(truth, est, board)
    assert r["kind"] == "coverage_miss"
    assert r["t"] == 90  # t_if_covered still says what we thought it was worth
    assert r["dev_b"] is None  # b = 0 is an abstention, not a small error


def test_items_with_no_evidence_are_dropped_rather_than_scored_as_zero():
    truth = {7: {"1": {"t_lo": 0.0, "t_hi": None, "charges": []}}}
    est = {7: {1: {"index": 1, "covered": True, "related": True, "t_mid": 50}}}
    assert D.rows(truth, est, {7: {}}) == []


def test_summary_counts_abstentions_instead_of_hiding_them():
    truth = {7: {"1": {"t_lo": 100.0, "t_hi": 200.0, "charges": []},
                 "2": {"t_lo": 100.0, "t_hi": 200.0, "charges": []}}}
    est = {7: {1: {"index": 1, "covered": True, "related": True, "t_mid": 150},
               2: {"index": 2, "covered": True, "related": True, "t_mid": 150}}}
    board = {7: {1: {"index": 1, "charge_price": 150.0, "acceptance_limit": 150.0},
                 2: {"index": 2, "charge_price": 150.0, "acceptance_limit": 0.0}}}
    s = D.summarise(D.rows(truth, est, board))
    assert s["b"]["n"] == 1 and s["b"]["zero"] == 1 and s["b"]["inside"] == 1
    assert s["objective"] == 0.0  # every distance that exists is zero


def test_parse_games_accepts_ranges_and_lists():
    assert D.parse_games("14-17") == [14, 15, 16, 17]
    assert D.parse_games("3,7,19") == [3, 7, 19]
    assert D.parse_games(None) is None


def test_euro_gap_is_zero_inside_the_bracket():
    assert D.euro_gap(100.0, 100.0, 200.0) == 0.0
    assert D.euro_gap(150.0, 100.0, 200.0) == 0.0


def test_euro_gap_is_the_signed_distance_to_the_bound_that_broke():
    assert D.euro_gap(60.0, 100.0, 200.0) == -40.0   # under by 40 EUR
    assert D.euro_gap(250.0, 100.0, 200.0) == 50.0   # over by 50 EUR
    assert D.euro_gap(200.0, 100.0, 200.0) == 0.0    # t_hi is exclusive: over by 0


def test_euro_gap_agrees_in_sign_with_the_log_distance():
    for x in (10.0, 99.0, 100.0, 150.0, 200.0, 4_000.0):
        assert math.copysign(1, D.euro_gap(x, 100.0, 200.0)) == \
               math.copysign(1, D.censored(x, 100.0, 200.0))


def test_euro_gap_has_no_value_for_an_abstention():
    assert D.euro_gap(0.0, 100.0, 200.0) is None
