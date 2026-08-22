from c2f.ensemble import aggregate


def it(i, cov, lo=0, mid=0, hi=0, ifc=0):
    return {"index": i, "covered": cov, "related": True, "t_low": lo, "t_mid": mid, "t_high": hi, "t_if_covered": ifc}


def test_majority_and_median():
    outs = [
        {"items": [it(1, True, 100, 200, 300), it(2, False, ifc=500)]},
        {"items": [it(1, True, 120, 250, 350), it(2, True, 400, 450, 500)]},
        {"items": [it(1, False, ifc=200), it(2, False, ifc=700)]},
    ]
    agg = {x["index"]: x for x in aggregate(outs)["items"]}
    assert agg[1]["covered"] is True and agg[1]["t_mid"] == 225 and agg[1]["t_low"] == 110
    assert agg[2]["covered"] is False and agg[2]["t_mid"] == 0
    assert agg[2]["t_if_covered"] == 500  # median of 500, 700, 450


def test_tie_is_not_covered():
    outs = [{"items": [it(1, True, 1, 2, 3)]}, {"items": [it(1, False, ifc=9)]}]
    assert aggregate(outs)["items"][0]["covered"] is False


def test_single_vote_passthrough():
    outs = [{"items": [it(3, True, 10, 20, 30)]}]
    a = aggregate(outs)["items"][0]
    assert a["index"] == 3 and a["covered"] and a["t_mid"] == 20
