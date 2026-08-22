"""Play one game:  pixi run python -m c2f.run GAME_ID [--no-submit] [--case-dir DIR]

IN (get_case.sh) -> EXTRACT -> DIGEST -> MODEL (fast, then full once) -> PRICE -> OUT.

Two passes, no votes. The fast pass is insurance, not a vote: its rows are submitted the
moment they land so that a slow or failed full pass can never leave us with nothing, and
the full pass overwrites them when it arrives (last write wins on the server). The fast
answer is NOT aggregated into the full one - over six logged games the two agreed on 70 of
82 coverage calls and split the other 12 evenly, so voting bought latency, not accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path

from c2f import llm, policy
from c2f.extract import load_case
from c2f.price import price_all
from c2f.submit import ROOT, fetch_case, submit

DEADLINE_S = 53.0  # clock restarts after decrypt (~1-3 s), server closes at 60 s after key release
FAST_TIMEOUT_S = 45.0
MIN_MODEL_S = 10.0  # never give the full pass less than this, however long the digest took
DIGEST_WAIT_S = 10.0  # how long the full pass waits for c2f.policy (~3 s typical) before going without


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
    ap.add_argument("--no-digest", action="store_true", help="skip the policy digest pass")
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
    log(f"{len(case['items'])} parsed line item(s), {len(case['images'])} image(s)", t0)
    record["case"] = {k: v for k, v in case.items() if k != "images"}
    record["case"]["n_images"] = len(case["images"])
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

    # ---- MODEL: fast pass for safety, then ONE full pass that overwrites it. No votes.
    full_model = os.environ.get("C2F_MODEL")
    fast_model = os.environ.get("C2F_FAST_MODEL") or (
        "claude-sonnet-5" if os.environ.get("ANTHROPIC_API_KEY") else "gpt-5-mini"
    )

    def run_model(model: str | None, timeout: float, strict: bool, c: dict):
        return llm.estimate(c, timeout=timeout, model=model, strict=strict)

    tags: dict = {}
    ex = ThreadPoolExecutor(max_workers=3)
    try:
        # Digest and fast pass start together; the fast pass is the safety net and must not wait
        # for the digest. dict(case) so attaching the digest cannot mutate a prompt already built.
        dig = None if args.no_digest else ex.submit(policy.build, case_dir)
        if not args.no_fast:
            tags[ex.submit(run_model, fast_model, FAST_TIMEOUT_S, True, dict(case))] = "fast"

        if dig is not None:
            try:
                digest_txt, record["digest"] = dig.result(timeout=DIGEST_WAIT_S)
            except FuturesTimeout:
                digest_txt, record["digest"] = None, {"error": f"not ready within {DIGEST_WAIT_S}s"}
            if digest_txt:
                case["policy_digest"] = digest_txt
                record["case"]["policy_digest"] = digest_txt
            log(f"policy digest: {record['digest']}", t0)
            save()

        budget = max(MIN_MODEL_S, DEADLINE_S - (time.time() - t0))
        tags[ex.submit(run_model, full_model, budget, False, case)] = "full"

        pending, full_done = set(tags), False
        while pending and not full_done:
            remaining = DEADLINE_S - (time.time() - t0)
            if remaining <= 0:
                log("deadline reached, stop waiting for the model", t0)
                break
            done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            # if both landed in the same batch, submit fast first so full is the last write
            for f in sorted(done, key=lambda f: tags[f] == "full"):
                tag = tags[f]
                try:
                    out, meta = f.result()
                except Exception as e:  # noqa: BLE001
                    record[f"model_{tag}_error"] = str(e)
                    log(f"model [{tag}] failed: {e}", t0)
                    continue
                record[f"model_{tag}"] = {"meta": {k: v for k, v in meta.items() if k != "raw"}, "output": out}
                record["estimate"] = out  # full runs last, so it wins; fast is the fallback
                log(f"model [{tag}] {meta['model']} answered in {meta['seconds']}s", t0)
                if tag == "full":
                    full_done = True
                rows = price_all(merge_estimates(case, out))
                record[f"priced_{tag}"] = rows
                do_submit(rows, tag)
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
