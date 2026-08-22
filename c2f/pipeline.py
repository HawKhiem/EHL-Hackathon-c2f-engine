"""Orchestrates one case end to end: (wait for key +) decrypt -> extract -> two LLM calls
(coverage, price) -> write cases/case_NN/estimate.json. Every stage logs its elapsed time so
it's obvious where the 60s submission budget goes."""
from __future__ import annotations

import concurrent.futures
import json
import time

from c2f import extract, game, history, llm

_t0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _t0:6.1f}s] {msg}", flush=True)


def estimate(game_id: int) -> list[dict]:
    log(f"game {game_id}: waiting for key / decrypting...")
    case_dir = game.decrypt(game_id)
    log(f"decrypted -> {case_dir}")

    case = extract.read_case(case_dir)
    hist = history.load()
    log(f"extracted invoice ({len(case.invoice_text)} chars), {len(hist)} history rows")

    # Independent calls, run concurrently - each is a separate 45s-timeout API call, and the
    # game only allows 60s to submit, so paying for them one after another is wasted budget.
    log("calling LLM (coverage + price in parallel)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        coverage_future = pool.submit(llm.estimate_coverage, case)
        prices_future = pool.submit(llm.estimate_prices, case.invoice_text, hist)
        coverage = {row["index"]: row for row in coverage_future.result()}
        log(f"coverage done ({len(coverage)} items)")
        prices = {row["index"]: row for row in prices_future.result()}
        log(f"prices done ({len(prices)} items)")

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
