from c2f.run import merge_estimates
from c2f.validate import invalid_indices, validate_items


def test_merge_fills_missing_and_orders_by_invoice():
    case = {"items": [{"index": 1}, {"index": 2}, {"index": 3}]}
    out = {"items": [{"index": 3, "covered": True}, {"index": "1", "covered": True}]}
    est = merge_estimates(case, out)
    assert [e["index"] for e in est] == [1, 2, 3]
    assert est[1]["covered"] is False and "not priced" in est[1]["reason"]


def test_merge_placeholder_is_a_valid_uncovered_item():
    """A line no chunk has reached is 0/0, and 0/0 must PASS validation.

    A bare stub has no t_low/t_mid/t_high, so c2f.validate calls it invalid and c2f.run
    reports it as a fatal item-level error. On a 39-item case split into four chunks that
    is ~30 spurious FATAL lines per chunk inside a 60 s window, hiding any real one.
    """
    est = merge_estimates({"items": [{"index": i} for i in range(1, 6)]},
                          {"items": [{"index": 1, "covered": True, "related": True,
                                      "t_low": 1, "t_mid": 2, "t_high": 3}]})
    assert invalid_indices(validate_items({"items": est})) == set()


def test_merge_without_parse_trusts_model_order():
    out = {"items": [{"index": 2}, {"index": 1}]}
    assert [e["index"] for e in merge_estimates({"items": []}, out)] == [1, 2]


def test_merge_labels_items_from_the_case_so_pricing_gets_a_category():
    """price.bias_for falls back to the flat global bias on a blank description."""
    case = {"items": [], "item_labels": {"1": "Skilled worker hours"}}
    est = merge_estimates(case, {"items": [{"index": 1, "covered": True}]})
    assert est[0]["_description"] == "Skilled worker hours"


def test_merge_refreshes_a_blank_label_on_a_stored_estimate():
    """make backtest reprices logs from games 27-29, which stored "_description": "" verbatim.

    setdefault kept that blank, so a repriced round stayed on the global bias no matter what
    the case could tell it.
    """
    case = {"items": [], "item_labels": {"1": "Skilled worker hours"}}
    stored = {"items": [{"index": 1, "covered": True, "_description": ""}]}
    assert merge_estimates(case, stored)[0]["_description"] == "Skilled worker hours"
