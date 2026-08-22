from c2f.truth import payout_bounds


def _tx(issuer, reviewer, item, accepted, amount):
    return {"issuer": issuer, "reviewer": reviewer, "line_item_index": item, "accepted": accepted, "amount": amount}


def test_payout_bounds_from_rejections():
    tx = [
        _tx("A", "X", 1, True, 765.0),   # A charges 765
        _tx("A", "Y", 1, False, 765.0),  # rejected but paid -> fair -> t >= 765
        _tx("B", "X", 1, True, 900.0),   # B charges 900
        _tx("B", "Y", 1, False, 0.0),    # rejected, paid 0 -> fraud -> t < 900
        _tx("C", "X", 2, True, 45.0),
        _tx("C", "Y", 2, False, 0.0),    # t < 45
        _tx("D", "X", 3, True, 10.0),    # never rejected: no info
    ]
    assert payout_bounds(tx) == {1: (765.0, 900.0), 2: (0.0, 45.0), 3: (0.0, None)}
