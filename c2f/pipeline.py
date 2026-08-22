"""Orchestrates one case end to end: decrypt -> extract -> two LLM calls (coverage, price) ->
write cases/case_NN/estimate.json."""
from __future__ import annotations

import concurrent.futures
import json

from c2f import extract, game, history, llm


def estimate(game_id: int) -> list[dict]:
    case_dir = game.decrypt(game_id)
    case = extract.read_case(case_dir)
    hist = history.load()

    # Independent calls, run concurrently - each is a separate 45s-timeout API call, and the
    # game only allows 60s to submit, so paying for them one after another is wasted budget.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        coverage_future = pool.submit(llm.estimate_coverage, case)
        prices_future = pool.submit(llm.estimate_prices, case.invoice_text, hist)
        coverage = {row["index"]: row for row in coverage_future.result()}
        prices = {row["index"]: row for row in prices_future.result()}

    items = []
    for index in sorted(set(coverage) | set(prices)):
        cov = coverage.get(index, {})
        pr = prices.get(index, {})
        items.append(
            {
                "index": index,
                "description": cov.get("description") or pr.get("description") or "",
                "covered": cov.get("covered", False),
                "t_low": pr.get("t_low", 0.0),
                "t_high": pr.get("t_high"),
            }
        )

    out_path = case_dir / "estimate.json"
    out_path.write_text(json.dumps(items, indent=1), encoding="utf-8")
    return items
