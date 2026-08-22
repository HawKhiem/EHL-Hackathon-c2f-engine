from c2f.run import merge_estimates


def test_merge_fills_missing_and_orders_by_invoice():
    case = {"items": [{"index": 1}, {"index": 2}, {"index": 3}]}
    out = {"items": [{"index": 3, "covered": True}, {"index": "1", "covered": True}]}
    est = merge_estimates(case, out)
    assert [e["index"] for e in est] == [1, 2, 3]
    assert est[1]["covered"] is False and "MISSING" in est[1]["reason"]


def test_merge_without_parse_trusts_model_order():
    out = {"items": [{"index": 2}, {"index": 1}]}
    assert [e["index"] for e in merge_estimates({"items": []}, out)] == [1, 2]
