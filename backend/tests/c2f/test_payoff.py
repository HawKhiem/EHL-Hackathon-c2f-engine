"""The payoff table. Anchored on the worked example in the brief."""

from __future__ import annotations

import pytest

from app.c2f.decision.payoff import score_field, score_item, transaction


def test_the_briefs_worked_example_reproduces_exactly():
    """Three teams, one windshield, t=100. The brief states the answer, so we check it.

    If this fails, the payoff module is wrong - not the numbers below.
    """
    field = {
        "Alpha Squad": (100.0, 130.0),
        "Beta Dynamics": (150.0, 90.0),
        "Delta Strategies": (100.0, 110.0),
    }
    scores = score_field(field, threshold=100.0)

    assert scores["Alpha Squad"].received == 200.0
    assert scores["Alpha Squad"].paid == 100.0
    assert scores["Alpha Squad"].net == 100.0

    assert scores["Delta Strategies"].received == 200.0
    assert scores["Delta Strategies"].paid == 100.0
    assert scores["Delta Strategies"].net == 100.0

    assert scores["Beta Dynamics"].received == 0.0
    assert scores["Beta Dynamics"].paid == 300.0
    assert scores["Beta Dynamics"].net == -300.0


def test_the_field_is_zero_sum_apart_from_lawyer_penalties():
    field = {"a": (100.0, 130.0), "b": (150.0, 90.0), "c": (100.0, 110.0)}
    scores = score_field(field, threshold=100.0)
    # Two wrongful rejects at 100 each burn 0.5 * 100 twice.
    assert sum(s.net for s in scores.values()) == pytest.approx(-100.0)


# ---------- the four cells of the table ----------


def test_fair_and_accepted_pays_the_charge():
    result = transaction(80.0, limit=100.0, threshold=100.0)
    assert (result.issuer_receives, result.insurer_pays) == (80.0, 80.0)
    assert result.label == "fair, accepted"


def test_fair_and_rejected_still_pays_the_issuer_plus_the_lawyer():
    """The result the whole strategy hangs on: rejection does not save the money."""
    result = transaction(80.0, limit=50.0, threshold=100.0)
    assert result.issuer_receives == 80.0
    assert result.insurer_pays == 120.0
    assert result.label == "wrongful reject"


def test_fraud_and_accepted_pays_up_to_the_cap():
    result = transaction(500.0, limit=600.0, threshold=100.0, cap=400.0)
    assert result.issuer_receives == 400.0
    assert result.insurer_pays == 400.0
    assert result.label == "fraud, accepted"


def test_fraud_and_rejected_moves_nothing():
    result = transaction(500.0, limit=100.0, threshold=100.0)
    assert (result.issuer_receives, result.insurer_pays) == (0.0, 0.0)
    assert result.label == "rightful reject"


def test_the_boundaries_are_inclusive():
    """`a <= t` is fair and `a <= b` is accepted, both inclusive."""
    assert transaction(100.0, limit=100.0, threshold=100.0).fair
    assert transaction(100.0, limit=100.0, threshold=100.0).accepted


def test_the_default_cap_is_four_times_the_threshold():
    assert transaction(1_000.0, limit=2_000.0, threshold=100.0).issuer_receives == 400.0


def test_an_uncovered_item_makes_every_positive_charge_fraud():
    assert not transaction(0.01, limit=100.0, threshold=0.0).fair
    # And the cap floor is zero, so accepting it moves nothing either.
    assert transaction(50.0, limit=100.0, threshold=0.0).issuer_receives == 0.0


# ---------- the issuer's charge is independent of the opponent's limit ----------


def test_a_fair_charge_collects_the_same_from_every_opponent():
    """R1 in docs/DESIGN.md, stated as money rather than algebra."""
    generous = score_item(100.0, 100.0, [(100.0, 10_000.0)], threshold=100.0)
    stingy = score_item(100.0, 100.0, [(100.0, 0.0)], threshold=100.0)
    assert generous.received == stingy.received == 100.0
    # The stingy opponent's own charge still gets accepted by us, so we pay it.
    assert generous.paid == stingy.paid == 100.0


def test_a_zero_limit_pays_the_penalty_to_every_opponent():
    """Why `b = 0` is the most expensive default available, not a safe one."""
    field = [(100.0, 100.0)] * 5
    assert score_item(100.0, 0.0, field, threshold=100.0).paid == 750.0


# ---------- the case 0 comparison, as a regression ----------


def test_case_0_scoring_ranks_the_strategies_we_expect():
    """t=420 on case 0. Both a low `b` and a low `a` cost money, `b` costs more.

    Numbers here are the ones in docs/DESIGN.md section 12. If this test moves,
    that table is stale.
    """
    opponents = [(420.0, 420.0), (400.0, 450.0), (800.0, 900.0), (1200.0, 300.0), (0.0, 0.0)]
    net = lambda a, b: score_item(a, b, opponents, 420.0).net  # noqa: E731

    oracle = net(420.0, 420.0)
    confident = net(418.65, 420.53)
    tight = net(382.83, 413.99)
    missed_clause = net(659.33, 692.63)

    assert oracle == pytest.approx(1280.0)
    assert confident == pytest.approx(1273.25)
    assert tight == pytest.approx(884.15)
    assert missed_clause < 0
    # A correct belief is worth more than 99% of knowing t outright.
    assert confident / oracle > 0.99
