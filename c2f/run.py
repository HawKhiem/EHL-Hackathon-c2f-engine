"""Play one game:  pixi run python -m c2f.run GAME_ID [--no-submit] [--case-dir DIR]

IN (get_case.sh) -> EXTRACT -> MODEL (fast, then full once) -> PRICE -> OUT.

Two passes, no votes. The fast pass is insurance, not a vote: its rows are submitted the
moment they land so that a slow or failed full pass can never leave us with nothing, and
the full pass overwrites them when it arrives (last write wins on the server). The fast
estimate is NOT averaged or merged into the full one's numbers - over six logged games the
two agreed on 70 of 82 coverage calls and split the other 12 evenly, so voting bought
latency, not accuracy. It is used only as a cheap disagreement check: c2f.price treats a
coverage split or a wide t_mid gap between the two passes as a reason to price the item
conservatively (see c2f/price.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from c2f import llm
from c2f.extract import load_case
from c2f.price import price_all
from c2f.submit import ROOT, fetch_case, submit
from c2f.validate import invalid_indices, validate_items

DEADLINE_S = 53.0  # clock restarts after decrypt (~1-3 s), server closes at 60 s after key release
FAST_TIMEOUT_S = 45.0
MIN_MODEL_S = 10.0  # never give the full pass less than this

# Above this many line items the full pass is split into parallel calls of at most this size.
# Games 10 and 15 both shipped a fast-pass-only answer because ONE long full call missed the
# deadline and its whole result was lost - 6 items with 46 s left, and 29 items with 28 s left.
# Split, the round's latency is the slowest chunk instead of the whole invoice, and a chunk that
# misses costs only its own items: the rest keep their refined prices.
CHUNK_ITEMS = 8
MAX_CHUNKS = 4  # more parallel calls than this starts competing for the same rate limit


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
    # The invoice wording rides along so c2f.price can pick a per-category bias. The model's
    # own output has no description field, and the category is where the estimate's bias
    # actually lives - one global multiplier is right for material and wrong for drying.
    descs = {int(it["index"]): it.get("description", "") for it in case.get("items", [])}
    for i in wanted:
        by_idx[i].setdefault("_description", descs.get(i, ""))
    return [by_idx[i] for i in wanted]


def value_of(est: dict) -> float:
    """A rough per-item stake, used only to decide what gets priced first."""
    for key in ("t_mid", "t_high", "t_if_covered", "t_low"):
        try:
            v = float(est.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if v > 0:
            return v
    return 0.0


def plan_chunks(indices: list[int], fast_out: dict | None) -> list[list[int]]:
    """Split the invoice into chunks, biggest-stake items first.

    Order matters as much as the split. Game 10's entire 63k loss was ONE line - a
    stolen watch worth 7,225 - so if only one chunk had come back it needed to be the
    one holding that item. When the fast pass has already landed we use its numbers to
    rank; otherwise invoice order is all we have.
    """
    if len(indices) <= CHUNK_ITEMS:
        return [list(indices)]
    ranked = list(indices)
    if fast_out:
        by_idx: dict[int, float] = {}
        for it in fast_out.get("items", []):
            try:
                by_idx[int(it["index"])] = value_of(it)
            except (KeyError, TypeError, ValueError):
                continue
        ranked.sort(key=lambda i: -by_idx.get(i, 0.0))
    n = min(MAX_CHUNKS, -(-len(ranked) // CHUNK_ITEMS))
    size = -(-len(ranked) // n)
    return [ranked[k : k + size] for k in range(0, len(ranked), size)]


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if f == f and abs(f) != float("inf") else 0.0


def resolve_invalid_items(items: list[dict], other_by_idx: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    """Replace every item that breaks the covered+related-must-be-positive contract (see
    c2f.validate), in this order: (1) the other model lane's valid estimate for that same
    index, (2) a positive t_if_covered from either lane. Anything still unresolved is
    reported as a fatal item-level error and priced as uncertain (covered=False) so pricing
    never receives the invalid all-zero-but-covered state (c2f.price.InvalidEstimateError) -
    but it is returned separately so the caller logs it as FATAL, not silently as an ordinary
    uncovered decision."""
    bad = invalid_indices(validate_items({"items": items}))
    if not bad:
        return items, []

    by_idx = {it["index"]: it for it in items}
    fatal: list[dict] = []
    for idx in sorted(bad):
        it = by_idx[idx]
        other = other_by_idx.get(idx)
        if other is not None and idx not in invalid_indices(validate_items({"items": [other]})):
            by_idx[idx] = {**other, "index": idx}
            continue
        guess = _num(it.get("t_if_covered")) or _num((other or {}).get("t_if_covered") if other else None)
        if guess > 0:
            by_idx[idx] = {**it, "t_low": guess, "t_mid": guess, "t_high": guess}
            continue
        fatal.append({"index": idx, "reason": "no valid valuation: item invalid, other lane has none, no t_if_covered"})
        by_idx[idx] = {**it, "covered": False, "related": False, "t_if_covered": 0}
    return [by_idx[i] for i in sorted(by_idx)], fatal


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
    fast_model = os.environ.get("C2F_FAST_MODEL") or "gpt-5.6-terra"

    def run_model(model: str | None, timeout: float, strict: bool, c: dict):
        return llm.estimate(c, timeout=timeout, model=model, strict=strict)

    def run_model_only(model: str | None, timeout: float, c: dict, only: list[int] | None, sweep: bool):
        """One full-pass chunk: whole case in the prompt, answer narrowed to `only`."""
        return llm.estimate(c, timeout=timeout, model=model, strict=False, only=only, sweep=sweep)

    tags: dict = {}
    # fast + up to MAX_CHUNKS full chunks run at once; a pool of 2 would serialise the
    # chunks and defeat the split. (The policy digest lane is gone, so no slot for it.)
    ex = ThreadPoolExecutor(max_workers=1 + MAX_CHUNKS)
    try:
        # Fast pass and full pass start together; the fast pass is the safety net and must
        # not wait for anything. dict(case) so the two passes never share a mutable dict.
        if not args.no_fast:
            tags[ex.submit(run_model, fast_model, FAST_TIMEOUT_S, True, dict(case))] = "fast"

        budget = max(MIN_MODEL_S, DEADLINE_S - (time.time() - t0))

        # Split the full pass. One long call is all-or-nothing: games 10 and 15 both lost the
        # entire full result to the deadline and shipped fast-pass prices. Chunked, the round's
        # latency is the slowest chunk and a chunk that misses costs only its own items.
        fast_landed = next((f for f in tags if tags[f] == "fast" and f.done()), None)
        fast_peek = None
        if fast_landed is not None:
            try:
                fast_peek = fast_landed.result(timeout=0)[0]
            except Exception:  # noqa: BLE001 - only used to order the chunks
                fast_peek = None
        parsed_idx = sorted({int(it["index"]) for it in case.get("items", [])})
        chunks = plan_chunks(parsed_idx, fast_peek) if parsed_idx else [None]
        for n, chunk in enumerate(chunks):
            tag = "full" if len(chunks) == 1 else f"full[{n + 1}/{len(chunks)}]"
            # chunk 0 also sweeps for POS numbers the parser missed
            tags[ex.submit(run_model_only, full_model, budget, dict(case), chunk, n == 0)] = tag
        if len(chunks) > 1:
            log(f"full pass split into {len(chunks)} chunks: {chunks}", t0)

        full_tags = {t for t in tags.values() if t.startswith("full")}
        full_items: dict[int, dict] = {}  # accumulates across chunks
        pending, full_seen = set(tags), set()
        while pending and full_seen != full_tags:
            remaining = DEADLINE_S - (time.time() - t0)
            if remaining <= 0:
                log("deadline reached, stop waiting for the model", t0)
                break
            done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            # if both landed in the same batch, submit fast first so full is the last write
            for f in sorted(done, key=lambda f: tags[f].startswith("full")):
                tag = tags[f]
                is_full = tag.startswith("full")
                try:
                    out, meta = f.result()
                except Exception as e:  # noqa: BLE001
                    record[f"model_{tag}_error"] = str(e)
                    log(f"model [{tag}] failed: {e}", t0)
                    if is_full:
                        full_seen.add(tag)  # a dead chunk must not stall the others
                    continue
                record[f"model_{tag}"] = {"meta": {k: v for k, v in meta.items() if k != "raw"}, "output": out}
                log(f"model [{tag}] {meta['model']} answered in {meta['seconds']}s", t0)
                for w in meta.get("validation_errors", []):
                    log(f"model [{tag}] INVALID: {w}", t0)

                if is_full:
                    full_seen.add(tag)
                    # Accumulate: each chunk refines its own items, and the accumulated set is
                    # re-priced and submitted in FULL every time. Submitting only the chunk's
                    # rows would rely on the server keeping omitted lines, and the handbook is
                    # ambiguous there ("upsert" vs "omitted line items use the game defaults").
                    # A wrong guess resets everything to 0/0, so we never test it under a clock.
                    for it in out.get("items", []):
                        try:
                            full_items[int(it["index"])] = it
                        except (KeyError, TypeError, ValueError):
                            continue
                    merged = {"items": list(full_items.values())}
                    fast_out = record.get("model_fast", {}).get("output")
                    # unrefined items fall back to the fast pass rather than to nothing
                    if fast_out:
                        for it in fast_out.get("items", []):
                            try:
                                i = int(it["index"])
                            except (KeyError, TypeError, ValueError):
                                continue
                            if i not in full_items:
                                merged["items"].append(it)
                    out, record["estimate"] = merged, merged
                    other_output = fast_out

                    # Every submission carries the FULL row set, so an item with no estimate
                    # from any source would go out at 0/0 - and 0/0 on a covered line pays the
                    # 1.5x penalty to every opponent. That is reachable now that the digest
                    # lane is gone and chunks start immediately: a late chunk can land before
                    # the fast pass, leaving the other items with nothing behind them. So hold
                    # the submission until some source covers every parsed item, unless nothing
                    # is still in flight - then partial beats silence.
                    have = set(full_items) | {
                        int(it["index"])
                        for it in (fast_out or {}).get("items", [])
                        if str(it.get("index", "")).lstrip("-").isdigit()
                    }
                    missing = [i for i in parsed_idx if i not in have]
                    if missing and pending:
                        log(
                            f"[{tag}] holding submission: {len(missing)} item(s) still unpriced "
                            f"by any pass ({missing[:6]}{'...' if len(missing) > 6 else ''})",
                            t0,
                        )
                        continue
                else:
                    record["estimate"] = out  # fast is the fallback until a chunk lands
                    other_output = None

                other_by_idx: dict[int, dict] = {}
                for it in (other_output or {}).get("items", []):
                    try:
                        other_by_idx[int(it["index"])] = it
                    except (KeyError, TypeError, ValueError):
                        continue
                resolved, fatal = resolve_invalid_items(merge_estimates(case, out), other_by_idx)
                for bad in fatal:
                    log(f"model [{tag}] FATAL: item {bad['index']}: {bad['reason']}", t0)
                if fatal:
                    record.setdefault("fatal_items", []).extend({**b, "tag": tag} for b in fatal)
                rows = price_all(resolved, other_output=other_output)
                record[f"priced_{tag}"] = rows
                do_submit(rows, tag)

        if pending:
            # deadline hit (or full landed first and fast is still out there) with a call still
            # in flight - it may finish after we've returned, but its result is lost either way.
            # Record which pass that was so a $0-fast-only submission is visible in the log, not
            # silently indistinguishable from "the full pass agreed with the fast one".
            still = sorted(tags[f] for f in pending)
            record["timed_out"] = still
            log(f"deadline reached with {still} still in flight - abandoning, using best submission so far", t0)
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
