"""Play one game:  pixi run python -m c2f.run GAME_ID [--no-submit] [--case-dir DIR]

IN (get_case.sh) -> EXTRACT -> MODEL (fast + full in parallel) -> PRICE -> OUT (PUT + runs/).
The fast model's answer is submitted as soon as it lands; the full model's answer
overwrites it if it arrives before the deadline. Last write wins on the server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from c2f import llm
from c2f.ensemble import aggregate
from c2f.extract import load_case
from c2f.price import price_all
from c2f.submit import ROOT, fetch_case, submit

DEADLINE_S = 53.0  # clock restarts after decrypt (~1-3 s), server closes at 60 s after key release
FULL_TIMEOUT_S = 50.0
FAST_TIMEOUT_S = 45.0
N_FULL = 3  # parallel full-model votes; override with C2F_N_FULL


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:5.1f}s] {msg}", flush=True)


def merge_estimates(case: dict, out: dict) -> list[dict]:
    """Model items, validated against the parsed invoice (if the parse succeeded)."""
    by_idx: dict[int, dict] = {}
    for it in out.get("items", []):
        try:
            it["index"] = int(it["index"])
            by_idx[it["index"]] = it
        except (KeyError, TypeError, ValueError):
            continue
    # union of parsed indices and model indices: never drop a line the model saw in the raw text
    wanted = sorted(set(it["index"] for it in case.get("items", [])) | set(by_idx))
    for i in wanted:
        if i not in by_idx:  # parsed but the model skipped it: price as unknown, log loudly
            by_idx[i] = {"index": i, "covered": False, "related": False, "reason": "MISSING FROM MODEL OUTPUT"}
    return [by_idx[i] for i in wanted]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("game_id", type=int)
    ap.add_argument("--no-submit", action="store_true")
    ap.add_argument("--case-dir", type=Path, help="skip get_case.sh and use this folder")
    ap.add_argument("--no-fast", action="store_true", help="skip the fast first pass")
    args = ap.parse_args(argv)
    try:
        llm.provider()  # fail before we touch the game if no LLM key is configured
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    t0 = time.time()
    record: dict = {"game_id": args.game_id, "started_at": t0, "submissions": []}
    dry = args.no_submit or args.case_dir is not None
    run_path = ROOT / "runs" / (f"dry_game_{args.game_id:02d}.json" if dry else f"game_{args.game_id:02d}.json")
    run_path.parent.mkdir(exist_ok=True)

    def save() -> None:
        run_path.write_text(json.dumps(record, indent=2, default=str))

    # ---- IN  (get_case.sh polls the key until the game opens; the 60 s clock starts then)
    if args.case_dir:
        case_dir = args.case_dir
    else:
        log("waiting for the key / decrypting ...", t0)
        case_dir = fetch_case(args.game_id)
        t0 = time.time()  # restart the clock: the window opened when the key appeared
        record["key_obtained_at"] = t0
    log(f"case at {case_dir}", t0)

    # ---- EXTRACT
    case = load_case(case_dir, args.game_id)
    record["case"] = {k: v for k, v in case.items() if k != "images"}
    record["case"]["n_images"] = len(case["images"])
    log(f"{len(case['items'])} parsed line item(s), {len(case['images'])} image(s)", t0)
    save()

    def do_submit(rows: list[dict], tag: str) -> None:
        entry = {"tag": tag, "at_s": round(time.time() - t0, 1), "rows": rows}
        if args.no_submit:
            entry["response"] = "skipped (--no-submit)"
        else:
            try:
                entry["response"] = submit(args.game_id, rows)
            except Exception as e:  # noqa: BLE001
                entry["error"] = str(e)
        record["submissions"].append(entry)
        save()
        log(f"submitted [{tag}]: " + ", ".join(f"#{r['index']} a={r['charge_price']} b={r['acceptance_limit']}" for r in rows), t0)

    # ---- MODEL: one fast pass + N_FULL full passes in parallel (ensemble)
    full_model = os.environ.get("C2F_MODEL")
    fast_model = os.environ.get("C2F_FAST_MODEL") or (
        "claude-sonnet-5" if os.environ.get("ANTHROPIC_API_KEY") else "gpt-5-mini"
    )
    n_full = int(os.environ.get("C2F_N_FULL", N_FULL))

    def run_model(tag: str, model: str | None, timeout: float, strict: bool):
        return tag, llm.estimate(case, timeout=timeout, model=model, strict=strict)

    futures = {}
    ex = ThreadPoolExecutor(max_workers=n_full + 1)
    full_outputs: list[dict] = []
    try:
        for i in range(n_full):
            futures[ex.submit(run_model, f"full{i}", full_model, FULL_TIMEOUT_S, False)] = f"full{i}"
        if not args.no_fast:
            futures[ex.submit(run_model, "fast", fast_model, FAST_TIMEOUT_S, True)] = "fast"

        pending = set(futures)
        while pending and len(full_outputs) < n_full:
            remaining = DEADLINE_S - (time.time() - t0)
            if remaining <= 0:
                log("deadline reached, stop waiting for the model", t0)
                break
            done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            for f in done:
                tag = futures[f]
                try:
                    _, (out, meta) = f.result()
                except Exception as e:  # noqa: BLE001
                    record[f"model_{tag}_error"] = str(e)
                    log(f"model [{tag}] failed: {e}", t0)
                    continue
                record[f"model_{tag}"] = {"meta": {k: v for k, v in meta.items() if k != "raw"}, "output": out}
                log(f"model [{tag}] {meta['model']} answered in {meta['seconds']}s", t0)
                if tag.startswith("full"):
                    full_outputs.append(out)
                    agg = aggregate(full_outputs)
                    record["ensemble"] = agg
                    rows = price_all(merge_estimates(case, agg))
                    record["priced_full"] = rows
                    do_submit(rows, f"full x{len(full_outputs)}")
                elif not full_outputs:
                    rows = price_all(merge_estimates(case, out))
                    record["priced_fast"] = rows
                    do_submit(rows, "fast")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    if not record["submissions"]:
        log("NO SUBMISSION MADE - both model passes failed", t0)
        save()
        return 1
    record["finished_at_s"] = round(time.time() - t0, 1)
    save()
    log(f"done. log: {run_path.relative_to(ROOT)}", t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
