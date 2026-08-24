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
from c2f.env import load_dotenv  # noqa: F401 - .env must be in os.environ before the constants below
from c2f.extract import case_labels, load_case
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
# One model, one pass (see llm.py): the call budget freed up by dropping the fast pass and the
# sol/luna lanes is spent on thinking instead - 10 items per call at a higher reasoning effort.
CHUNK_ITEMS = int(os.environ.get("C2F_CHUNK_ITEMS") or 10)  # n -> ceil(n/10) calls: 10 -> 1, 20 -> 2, 32 -> 4
MAX_CHUNKS = int(os.environ.get("C2F_MAX_CHUNKS") or 8)  # backstop (80 items); past that the calls fight the rate limit


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
        if i not in by_idx:
            # A POS number no chunk has priced yet. It must be a COMPLETE uncovered item, not
            # a bare stub: c2f.validate requires t_low/t_mid/t_high on every item, so a stub
            # is reported as a fatal invalid estimate and 30 of them bury the one real FATAL
            # in the log of a 39-item case. 0/0 is the intended ignorant placeholder (see the
            # submit-after-every-chunk note below), so say so explicitly and stay valid.
            by_idx[i] = {"index": i, "covered": False, "related": False, "t_low": 0, "t_mid": 0,
                         "t_high": 0, "t_if_covered": 0, "reason": "not priced by any chunk yet"}
    # The invoice wording rides along so c2f.price can pick a per-category bias. The model's
    # own output has no description field, and the category is where the estimate's bias
    # actually lives - one global multiplier is right for material and wrong for drying.
    # This must NOT read case["items"]: that list is empty since the invoice stopped being
    # line-parsed, and a blank description sends price.bias_for straight to the global bias.
    # Games 27 and 28 priced every line that way. case_labels reads the labels instead.
    descs = case_labels(case)
    # NOT setdefault: make backtest reprices STORED estimates, and the logs from games 27-29
    # carry "_description": "" already, which setdefault would keep.
    for i in wanted:
        if not by_idx[i].get("_description"):
            by_idx[i]["_description"] = descs.get(i, "")
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
    # One pass by default. The fast pass was insurance from when the full pass ran on a
    # slower model (gpt-5.6-sol, 7-35 s); now that everything is gpt-5.6-terra the full pass
    # lands in 2-7 s and the second call bought a duplicate answer, not safety. --fast brings
    # it back if a case ever runs long again.
    ap.add_argument("--fast", dest="fast", action="store_true", help="also run the fast safety pass first")
    ap.add_argument("--no-fast", dest="fast", action="store_false", help="skip the fast first pass (default)")
    ap.set_defaults(fast=False)
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

    # ---- STRATEGY v2: primary estimate + independent coverage and valuation audits in parallel.
    # Every role can produce a complete fallback board; later roles tighten b as they land.
    if os.environ.get("C2F_STRATEGY", "").lower() == "v2":
        from c2f import v2 as V2

        # The first completed role still lands the safety board immediately. Keep listening for
        # audit refinements until the real submission deadline; the backtest measured 31-51 s.
        v2_deadline = float(os.environ.get("C2F_V2_DEADLINE_S") or DEADLINE_S)
        memory = V2.live_memory()
        record["strategy"] = "v2"
        budget = max(MIN_MODEL_S, DEADLINE_S - (time.time() - t0))
        exv = ThreadPoolExecutor(max_workers=len(V2.ROLES))
        futs = {
            exv.submit(V2.call_v2, dict(case), memory, budget, role=role): role
            for role in V2.ROLES
        }
        outs: list[dict] = []
        try:
            pending = set(futs)
            while pending:
                remaining = min(DEADLINE_S, v2_deadline) - (time.time() - t0)
                if remaining <= 0:
                    log(f"v2 cutoff ({v2_deadline:.0f}s) reached with {len(pending)} sample(s) still out - keeping the board as is", t0)
                    break
                done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
                for f in done:
                    try:
                        out, secs = f.result()
                    except Exception as e:  # noqa: BLE001
                        log(f"v2 {futs[f]} failed: {e}", t0)
                        continue
                    outs.append(out)
                    log(f"v2 {futs[f]} answered in {secs}s", t0)
                    combined = V2.combine_samples(outs, case)
                    rows, dbg = V2.rows_v2(case, combined, memory)
                    record["estimate_v2"] = combined
                    record["v2_debug"] = {str(i): d for i, d in dbg.items()}
                    do_submit(rows, "v2[" + "+".join(o.get("_role", "?") for o in outs) + "]")
        finally:
            exv.shutdown(wait=False, cancel_futures=True)
        if not record["submissions"]:
            log("v2 produced nothing - falling back to the v1 pass", t0)
        else:
            record["finished_at_s"] = round(time.time() - t0, 1)
            save()
            log(f"done (v2). log: {run_path.relative_to(ROOT)}", t0)
            return 0

    # ---- MODEL: ONE pass. No votes, and no fast safety pass unless --fast is given.
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
    ex = ThreadPoolExecutor(max_workers=(1 if args.fast else 0) + MAX_CHUNKS)
    try:
        # Fast pass and full pass start together; the fast pass is the safety net and must
        # not wait for anything. dict(case) so the two passes never share a mutable dict.
        if args.fast:
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
        asked: dict[str, list[int]] = {}  # tag -> the POS numbers that chunk was asked for
        for n, chunk in enumerate(chunks):
            tag = "full" if len(chunks) == 1 else f"full[{n + 1}/{len(chunks)}]"
            # The LAST chunk sweeps for POS numbers the parser missed - not the first.
            # A sweeping chunk is not really a chunk: it is told to price every POS number in
            # the invoice that is absent from its list, so it answers for the whole invoice and
            # takes whole-invoice time. Measured on case 15 (29 items, chunk [1..10], two runs
            # each, run concurrently so load is equal): sweep 32-36 s returning 29 items,
            # no sweep 17-28 s returning 10. Carrying that ~12 s on chunk 0 put it on the
            # BIGGEST-stake items (plan_chunks ranks them first), so the lines worth most sat
            # at the 0/0 placeholder longest - and games 10 and 15 were both lost to the clock.
            # On the last, smallest, lowest-stake chunk the same insurance costs the least.
            tags[ex.submit(run_model_only, full_model, budget, dict(case), chunk,
                           n == len(chunks) - 1)] = tag
            asked[tag] = list(chunk) if chunk else list(parsed_idx)
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
                    # THIS is a missing item: a POS number we put in front of the model and
                    # asked it to price, that is absent from its answer. Distinct from a line
                    # a later chunk still owns, and the only one worth shouting about - the
                    # parse exists so that a silent skip cannot pass for a priced line.
                    answered = {int(it["index"]) for it in out.get("items", [])
                                if str(it.get("index", "")).lstrip("-").isdigit()}
                    skipped = [i for i in asked.get(tag, []) if i not in answered]
                    if skipped:
                        record.setdefault("skipped_by_model", []).extend(
                            {"tag": tag, "index": i} for i in skipped)
                        log(f"model [{tag}] SKIPPED {len(skipped)} item(s) it was asked to "
                            f"price: {skipped} - they stay at the 0/0 placeholder", t0)
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

                    # SUBMIT AFTER EVERY CHUNK. Last write wins on the server and only the
                    # state at close is scored, so an intermediate row that is still a
                    # placeholder costs nothing as long as a later chunk overwrites it - while
                    # a board that is never empty survives a crash, a hang, or a clock we
                    # misjudged. We used to hold until every parsed item had an estimate; with
                    # the fast pass gone that meant one slow chunk could ship NOTHING.
                    #
                    # The placeholder for an item no chunk has reached yet is 0/0, and 0/0 is
                    # the right ignorant guess: b=0 wrongly rejects fair charges at 0.5a extra
                    # each, while b=inf accepts fraud up to the cap c >= 4t. Rejecting is the
                    # cheaper mistake to make blind, so an unreached line is a safe line.
                    have = set(full_items) | {
                        int(it["index"])
                        for it in (fast_out or {}).get("items", [])
                        if str(it.get("index", "")).lstrip("-").isdigit()
                    }
                    missing = [i for i in parsed_idx if i not in have]
                    if missing:
                        log(
                            f"[{tag}] submitting with {len(missing)} item(s) still at the 0/0 "
                            f"placeholder ({missing[:6]}{'...' if len(missing) > 6 else ''}) - "
                            "a later chunk overwrites them",
                            t0,
                        )
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

    # ---- REFINE: median-of-k on whale items, when the clock allows. One sample of t_mid on
    # a dominant line swings the round (game 41: 8,000 vs 6,000 on the same prompt = +-65k),
    # so if the board is already safe (a submission exists) and >= 12 s remain, resample the
    # big items and submit once more - last write wins.
    remaining = DEADLINE_S - (time.time() - t0)
    est_now = record.get("estimate") or {}
    if record["submissions"] and remaining >= 12.0 and est_now.get("items"):
        try:
            refined = llm.resample_whales(case, est_now, timeout=remaining - 4.0, model=full_model)
            n_resampled = sum(1 for it in refined.get("items", []) if it.get("_n_samples", 0) > 1)
            if n_resampled:
                record["estimate"] = refined
                resolved, _ = resolve_invalid_items(merge_estimates(case, refined), {})
                do_submit(price_all(resolved, other_output=None), "refine")
                log(f"refined {n_resampled} whale item(s) with extra samples", t0)
        except Exception as e:  # noqa: BLE001 - refinement must never cost the round
            log(f"refine pass failed (keeping earlier submission): {e}", t0)

    # Last-ditch flush. The hold above waits for every parsed item to have an estimate before
    # submitting, which used to be safe because the fast pass had already put rows on the board.
    # With one pass there is nothing behind it: a deadline hit while holding would ship NOTHING.
    # Partial rows beat silence - the items a dead chunk never covered go out at 0/0 either way.
    if not record["submissions"] and (record.get("estimate") or {}).get("items"):
        log("deadline hit while holding - submitting the partial estimate rather than nothing", t0)
        try:
            resolved, _ = resolve_invalid_items(merge_estimates(case, record["estimate"]), {})
            do_submit(price_all(resolved, other_output=None), "partial")
        except Exception as e:  # noqa: BLE001 - a broken partial must not mask the real failure
            log(f"partial submission failed: {e}", t0)

    if not record["submissions"]:
        log("NO SUBMISSION MADE - the model pass failed", t0)
        save()
        return 1
    record["finished_at_s"] = round(time.time() - t0, 1)
    save()
    log(f"done. log: {run_path.relative_to(ROOT)}", t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
