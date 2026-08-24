from c2f.extract import item_quantities
from c2f.v2 import ROLE_INSTRUCTIONS, combine_samples, key_of, price_v2, rows_v2, scale_memory, user_message_v2


def test_coverage_prompt_requires_verbatim_quote_for_policy_exclusion():
    prompt = " ".join(ROLE_INSTRUCTIONS["coverage"].split())
    assert "adversarial COVERAGE auditor" in prompt
    assert "missing prerequisite" in prompt
    assert "four consecutive words verbatim" in prompt
    assert "Never paraphrase that quote" in prompt


def test_quantity_memory_and_independent_b_gate():
    invoice = """POS. DESCRIPTION AMOUNTUNIT TOTAL
1 Service technician hours 9 labor units
2 Tile installation
6 m²
INVOICE
"""
    assert item_quantities(invoice) == {1: (9.0, "labor units"), 2: (6.0, "m²")}
    assert scale_memory([(1_000.0, 1_200.0, 10.0, "hour")], 1.0, "hrs") == [(100.0, 120.0)]
    case = {"policy": "policy", "description": "damage", "invoice_text": invoice,
            "invoice_meta": {}, "items": [{"index": 1, "description": "Service technician hours",
                                             "quantity": 9.0, "unit": "labor units"}]}
    memory = {key_of("Service technician hours"): [
        (1_000.0, 1_200.0, 10.0, "hour", 12, 3, "Premium emergency technician hours",
         "Specialist work on a high-end heating system"),
        (100_000.0, 120_000.0, 10.0, "hour", 13, 4, "Luxury-brand specialist hours",
         "Work on a materially different collector-grade system"),
    ]}
    assert "<memory>" not in user_message_v2(case, memory, "primary")
    valuation_prompt = user_message_v2(case, memory, "valuation")
    assert "<memory>" in valuation_prompt and 'id="12:3"' in valuation_prompt
    assert "high-end heating system" in valuation_prompt

    estimate = {"items": [{"index": 1, "p_covered": 0.98, "p_accept": 0.98,
                            "q10": 90, "q50": 100, "q90": 110,
                            "coverage_supported": True, "value_supported": True,
                            "comparable_history_ids": []}]}
    _, debug = rows_v2(case, estimate, memory)
    assert debug[1]["mem"] is True  # normalized service history does not need brand identity

    watch_case = {"invoice_text": "", "items": [
        {"index": 1, "description": "Compensation for stolen watch", "quantity": 1.0, "unit": "pcs"}
    ]}
    watch_memory = {key_of("Compensation for stolen watch"): [
        (1_000.0, 1_200.0, 1.0, "pcs", 12, 3, "Compensation for budget watch", "Basic model"),
        (100_000.0, 120_000.0, 1.0, "pcs", 13, 4, "Compensation for luxury watch", "Collector model"),
    ]}
    estimate["items"][0]["comparable_history_ids"] = []
    _, debug = rows_v2(watch_case, estimate, watch_memory)
    assert debug[1]["mem"] is False
    estimate["items"][0]["comparable_history_ids"] = ["12:3"]
    _, debug = rows_v2(watch_case, estimate, watch_memory)
    assert debug[1]["mem"] is True and debug[1]["mu"] < 8  # unselected luxury watch ignored

    # Current-case coverage wins: paid history cannot turn 1% into an accepted line.
    a, b, debug = price_v2(
        {"p_covered": 0.01, "q50": 850, "coverage_supported": True, "value_supported": True},
        [(800.0, 1_000.0)],
    )
    assert a > 0 and b == 0 and debug["p_cov"] == 0.01

    # A large b needs both auditors; a concrete coverage failure still leaves the issuer's a.
    a, b, _ = price_v2(
        {"p_covered": 0.98, "q50": 17_000, "coverage_supported": False, "value_supported": True},
        None,
    )
    assert a > 0 and b == 0
    _, supported_b, _ = price_v2(
        {"p_covered": 0.98, "q50": 750, "coverage_supported": True, "value_supported": True},
        None,
    )
    assert 0 < supported_b < 750
    _, independently_audited_b, _ = price_v2(
        {"p_covered": 0.01, "p_accept": 0.98, "q50": 750,
         "coverage_supported": True, "value_supported": True},
        None,
    )
    assert independently_audited_b == supported_b

    # The exact decision boundary is a rejection, not inv_cdf(0), which raises StatisticsError.
    _, boundary_b, _ = price_v2(
        {"p_covered": 0.98, "p_accept": 2 / 3, "q50": 750,
         "coverage_supported": True, "value_supported": True},
        None,
    )
    assert boundary_b == 0


def test_specialists_merge_and_code_does_quantity_math():
    case = {"items": [{"index": 1, "description": "Service technician hours", "quantity": 9.0,
                       "unit": "labor units"}]}
    outs = [
        {"_role": "primary", "items": [{"index": 1, "p_covered": 0.98, "q10": 400,
                                           "q50": 500, "q90": 700, "reason": "covered leak"}]},
        {"_role": "coverage", "items": [{"index": 1, "p_covered": 0.97, "q10": 400,
                                            "q50": 500, "q90": 700,
                                            "coverage_supported": True, "reason": "necessary repair"}]},
        {"_role": "valuation", "items": [{"index": 1, "p_covered": 0.96, "q10": 600,
                                             "q50": 700, "q90": 800, "unit_q10": 70,
                                             "unit_q50": 80, "unit_q90": 90,
                                             "value_supported": True,
                                             "comparable_history_ids": ["12:3"],
                                             "history_reason": "same technician grade",
                                             "reason": "nine billed hours"}]},
    ]
    item = combine_samples(outs, case)["items"][0]
    assert (item["q10"], item["q50"], item["q90"]) == (630.0, 720.0, 810.0)
    assert item["p_covered"] == 0.98 and item["p_accept"] == 0.97
    assert item["coverage_supported"] is True and item["value_supported"] is True
    assert item["comparable_history_ids"] == ["12:3"]

    outs[2]["items"][0].update({"q10": 0, "q50": 0, "q90": 0,
                                  "unit_q10": 0, "unit_q50": 0, "unit_q90": 0})
    item = combine_samples(outs, case)["items"][0]
    assert (item["q10"], item["q50"], item["q90"]) == (400.0, 500.0, 700.0)
    assert item["value_supported"] is False


def test_policy_exclusion_requires_a_verbatim_policy_quote():
    case = {
        "policy": "Damage caused by ordinary wear and tear is not insured under this policy.",
        "items": [{"index": 1, "description": "Worn flooring"}],
    }

    def combined(quote, denial="policy_exclusion"):
        return combine_samples([{
            "_role": "coverage",
            "items": [{
                "index": 1,
                "p_covered": 0.05,
                "q10": 100,
                "q50": 200,
                "q90": 300,
                "coverage_supported": False,
                "coverage_denial": denial,
                "policy_quote": quote,
            }],
        }], case)["items"][0]

    exact = combined("ordinary wear and tear is not insured")
    assert exact["coverage_supported"] is False
    assert exact["policy_quote_verified"] is True

    invented = combined("gradual deterioration is always excluded")
    assert invented["coverage_supported"] is None
    assert invented["policy_quote_verified"] is False

    unrelated = combined("", "unrelated_or_upgrade")
    assert unrelated["coverage_supported"] is False
    assert unrelated["policy_quote_verified"] is None
