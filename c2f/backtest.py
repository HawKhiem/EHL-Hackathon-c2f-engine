"""Replay the CURRENT strategy on past games and score it against what the other teams actually did.

  pixi run python -m c2f.backtest              # re-price + re-score every stored replay; NO model call
  pixi run python -m c2f.backtest --replay     # call the model again for every old game, then score
  pixi run python -m c2f.backtest --replay 2 4 # call the model for these games only; others from store

The default never calls the model: each old game's stored model estimate (runs/backtest/game_NN.json)
is re-priced with the CURRENT price.py + calibration and re-scored, so pricing changes are evaluated
in seconds. --replay is for changes upstream of pricing (prompt, extract, policy digest).

Per game:  extract -> digest -> model (same code path as a live run) -> price -> simulate every pairing
against the real opponents of that round, using the public leaderboard:
  - each opponent's charge a_j per item and whether it was fair (from payouts),
  - each opponent's accept limit b_j as an interval [b_lo, b_hi) (from what they accepted/rejected),
  - the fair value t as an interval [t_lo, t_hi) (largest charge proven fair, smallest proven fraud).
Where an interval leaves the outcome open we score two scenarios: PESSIMISTIC (t = t_lo, opponents
reject whenever they could) and OPTIMISTIC (t just under t_hi, opponents accept whenever they could).
Our rank = where our replayed net would have put us in that round's standings.

THE BAR IS MONEY, AND ONLY MONEY. EXPECTED = midpoint of the two scenarios; a change is a SUCCESS
when the expected replay is PROFITABLE (net > 0) in more than half of the old games and the total
expected net is positive. Rank is reported but does not gate: a steady 3rd place every round while
in profit is exactly the target outcome, so we never fail a strategy for not topping the table.
The population is the LAST 5 completed games whose case is decrypted locally (WINDOW), not just the
games replayed in this invocation: games you don't name are re-scored from their stored replay (no
model call), games with no stored replay count as a failure and are listed as missing.

This is THE evaluation entry point. It uses the other tools as components:
  feedback.digest  -> opponents' charges, accept/reject, payout-based t bounds
  truth.infer      -> tighter t bounds from the matchup cells (cached in runs/truth_game_NN.json)
  calibration.json -> reported alongside, so a result is tied to the calibration it ran with
Writes runs/backtest/game_NN.json and runs/backtest/summary.json (+ a table on stdout). Nothing
enforces the verdict - run this when a change is worth checking and read the table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

from c2f import llm
from c2f.extract import load_case
from c2f.feedback import B, DEFAULT_TEAM, digest
from c2f.policy import attach as attach_policy_digest
from c2f.price import price_all
from c2f.run import merge_estimates
from c2f.submit import ROOT
from c2f import truth as truth_mod

OUT = ROOT / "runs" / "backtest"
INF = float("inf")
WINDOW = 5  # the verdict is over the LAST 5 completed games with a decrypted case
RANK_TARGET = 3  # top-3 is the outcome we aim for; reported only - the verdict gates on money


# ----------------------------------------------------------------------------- evidence
def t_bounds(game_id: int, d: dict) -> dict:
    """Tightest fair-value bounds per item: payout evidence (feedback.digest) merged with the
    matchup-cell inference (truth.py). truth results are cached in runs/truth_game_NN.json."""
    p = ROOT / "runs" / f"truth_game_{game_id:02d}.json"
    tr: dict = {}
    if p.exists():
        tr = json.loads(p.read_text())
    else:
        try:
            tr = {str(k): v for k, v in truth_mod.infer(game_id).items()}
            p.write_text(json.dumps(tr, indent=1))
        except Exception as e:  # noqa: BLE001 - truth is best-effort; payout bounds still apply
            print(f"   (truth inference unavailable for game {game_id}: {e.__class__.__name__})")
    src = {}
    for i in d["items"]:
        lo, hi = d["t_lo"][i], d["t_hi"][i]
        v = tr.get(str(i))
        if v:
            lo = max(lo, float(v.get("t_lo") or 0.0))
            if v.get("t_hi") is not None:
                hi = min(hi, float(v["t_hi"]))
        d["t_lo"][i], d["t_hi"][i] = lo, hi
        src[i] = "truth+payout" if v else "payout"
    return src


# ----------------------------------------------------------------------------- replay
def replay(game_id: int) -> dict:
    """One model call, exactly as run.py plays it."""
    case_dir = ROOT / "cases" / f"case_{game_id:02d}"
    case = load_case(case_dir, game_id)
    attach_policy_digest(case, case_dir)  # same prompt the live run would build
    t0 = time.time()
    out, _meta = llm.estimate(case, timeout=60, strict=False)
    rows = price_all(merge_estimates(case, out))
    return {"rows": rows, "estimate": out, "seconds": round(time.time() - t0, 1)}


def reprice(game_id: int, rep: dict) -> dict:
    """Price the stored model estimate with the current price.py + calibration; no model call."""
    est = rep.get("estimate") or rep.get("ensemble")
    if not est:
        return rep
    case = load_case(ROOT / "cases" / f"case_{game_id:02d}", game_id)
    return {**rep, "rows": price_all(merge_estimates(case, est)), "repriced": True}


# ----------------------------------------------------------------------------- scoring
def _fair_status(a: float, t_lo: float, t_hi: float) -> bool | None:
    """True = proven fair, False = proven fraud, None = open."""
    if a <= 0:
        return True
    if a <= t_lo + 1e-9:
        return True
    if a >= t_hi - 1e-9:
        return False
    return None


def score(game_id: int, rows: list[dict], d: dict, us: str) -> dict:
    ours = {r["index"]: (r["charge_price"], r["acceptance_limit"]) for r in rows}
    opp = [t for t in d["teams"] if t != us]
    res = {"pessimistic": {"income": 0.0, "cost": 0.0}, "optimistic": {"income": 0.0, "cost": 0.0}}
    open_income = open_cost = 0
    per_item = {}
    for i in d["items"]:
        a, b = ours.get(i, (0.0, 0.0))
        t_lo, t_hi = d["t_lo"][i], d["t_hi"][i]
        fair_ours = _fair_status(a, t_lo, t_hi)
        inc = {"pessimistic": 0.0, "optimistic": 0.0}
        cost = {"pessimistic": 0.0, "optimistic": 0.0}
        for j in opp:
            # ---- we issue a to reviewer j
            b_lo, b_hi = d["b_lo"][j].get(i, 0.0), d["b_hi"][j].get(i, INF)
            acc = True if a <= b_lo + 1e-9 else (False if a >= b_hi - 1e-9 else None)
            for sc in ("pessimistic", "optimistic"):
                fair = fair_ours if fair_ours is not None else (sc == "optimistic")
                accepted = acc if acc is not None else (sc == "optimistic")
                if fair:
                    inc[sc] += a
                elif accepted:
                    inc[sc] += a  # cap assumed not binding
            if fair_ours is None or acc is None:
                open_income += 1
            # ---- j issues a_j to us
            a_j = d["issued"][j].get(i)
            if a_j is None or a_j <= 0:
                # unknown charge (rejected by everyone, paid 0 => fraud, a_j > 0 unknown) or a_j = 0
                # we only pay it if our b >= a_j; a_j unknown but every team rejected it, assume we do too
                continue
            fair_j = d["fair"][j].get(i)
            if fair_j is None:
                fair_j = _fair_status(a_j, t_lo, t_hi)
            we_accept = a_j <= b + 1e-9
            for sc in ("pessimistic", "optimistic"):
                fair = fair_j if fair_j is not None else (sc == "pessimistic")
                if fair:
                    cost[sc] += a_j if we_accept else 1.5 * a_j
                elif we_accept:
                    cost[sc] += a_j
            if fair_j is None:
                open_cost += 1
        for sc in res:
            res[sc]["income"] += inc[sc]
            res[sc]["cost"] += cost[sc]
        per_item[i] = {"a": a, "b": b, "t_lo": t_lo, "t_hi": None if t_hi == INF else t_hi, "fair": fair_ours,
                       "income": inc, "cost": cost}
    for sc in res:
        res[sc]["net"] = round(res[sc]["income"] - res[sc]["cost"], 2)
        res[sc]["income"] = round(res[sc]["income"], 2)
        res[sc]["cost"] = round(res[sc]["cost"], 2)
    return {"scenarios": res, "open_pairings": {"income": open_income, "cost": open_cost}, "items": per_item}


def actual_nets() -> tuple[list[int], dict[str, dict[int, float]]]:
    m = requests.get(f"{B}/matrix?page=1&game_limit=1000", timeout=20).json()
    gids = m["game_ids"]
    nets = {row["team_name"]: {g: c for g, c in zip(gids, row["cells"])} for row in m["items"]}
    return gids, nets


def rank_of(net: float, others: list[float]) -> int:
    return 1 + sum(1 for x in others if x > net)


def old_games(gids: list[int], window: int = WINDOW) -> list[int]:
    """The population the verdict is over: the last `window` completed games whose case is
    decrypted locally (older games drift - other teams change strategy between rounds)."""
    have = sorted(g for g in gids if (ROOT / "cases" / f"case_{g:02d}" / "policy.txt").exists())
    return have[-window:]


def verdict(games: dict, old: list[int]) -> dict:
    """Totals + SUCCESS flag over the full set of old games. `games` holds the per-game rows we have
    (replayed now or re-scored from the store); old games without a row count as a failure.

    The bar is MONEY, and only money: a game counts when the expected replay net is positive.
    Rank is reported (`top_ranked`, `good`, `wins_exp`) but does NOT gate the verdict - placing
    3rd every round while in profit is the target outcome, so a strategy that pays in every game
    passes even if it never tops the table.
    """
    rows = [games[g] for g in old if g in games]
    missing = [g for g in old if g not in games]
    n = len(old)
    paid = sum(1 for v in rows if v["exp_net"] > 0)
    top = sum(1 for v in rows if v["exp_rank"] <= RANK_TARGET)
    good = sum(1 for v in rows if v["exp_net"] > 0 and v["exp_rank"] <= RANK_TARGET)
    return {
        "actual": sum(v["actual_net"] for v in rows), "pessimistic": sum(v["pess_net"] for v in rows),
        "expected": sum(v["exp_net"] for v in rows), "optimistic": sum(v["opt_net"] for v in rows),
        "profitable": paid, "top_ranked": top, "good": good, "rank_target": RANK_TARGET,
        "wins_pess": sum(1 for v in rows if v["pess_rank"] == 1),
        "wins_exp": sum(1 for v in rows if v["exp_rank"] == 1),
        "wins_opt": sum(1 for v in rows if v["opt_rank"] == 1),
        "n_games": n, "n_scored": len(rows), "missing": missing,
        "success": n > 0 and not missing and paid * 2 > n and sum(v["exp_net"] for v in rows) > 0,
    }


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*", type=int)
    ap.add_argument("--replay", action="store_true", help="call the model for the named games (default: all old games); otherwise stored estimates are re-priced")
    ap.add_argument("--no-llm", action="store_true", help=argparse.SUPPRESS)  # old spelling of the default
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    gids, nets = actual_nets()
    us = DEFAULT_TEAM
    old = old_games(gids)
    requested = args.games or old
    # every old game is scored: with --replay the requested ones call the model, everything else is
    # re-priced from its stored estimate (no model call)
    for g in requested:
        if g not in gids:
            print(f"{g:>4}  not completed yet, skipped")
        elif g not in old:
            print(f"{g:>4}  not in the last {WINDOW} decrypted games {old} (or not decrypted), skipped")
    plan = [(g, args.replay and g in requested) for g in old]
    summary = {"team": us, "games": {}, "old_games": old, "generated_at": time.time()}
    print(f"backtest as {us}: {len(old)} old games {old}, replaying {[g for g, r in plan if r]}\n")
    print(f"{'game':>4} {'items':>5} | {'actual net':>10} {'rank':>4} | {'replay pess':>11} {'rank':>4} | {'replay exp':>10} {'rank':>4} | {'replay opt':>10} {'rank':>4} | best team")
    for g, do_replay in plan:
        path = OUT / f"game_{g:02d}.json"
        if do_replay:
            rep = replay(g)
        elif path.exists():
            rep = reprice(g, json.loads(path.read_text())["replay"])
        else:
            print(f"{g:>4}  no stored replay - MISSING, counts as not won (run: make replay G={g})")
            continue
        d = digest(g)
        evidence = t_bounds(g, d)
        sc = score(g, rep["rows"], d, us)
        sc["evidence"] = evidence
        others = [nets[t][g] for t in nets if t != us]
        best = max(nets.items(), key=lambda kv: kv[1][g])
        actual = nets.get(us, {}).get(g, float("nan"))
        exp_net = (sc["scenarios"]["pessimistic"]["net"] + sc["scenarios"]["optimistic"]["net"]) / 2
        row = {
            "actual_net": actual,
            "actual_rank": rank_of(actual, others),
            "replay": rep,
            "score": sc,
            "pess_net": sc["scenarios"]["pessimistic"]["net"],
            "pess_rank": rank_of(sc["scenarios"]["pessimistic"]["net"], others),
            "opt_net": sc["scenarios"]["optimistic"]["net"],
            "opt_rank": rank_of(sc["scenarios"]["optimistic"]["net"], others),
            "exp_net": round(exp_net, 2),
            "exp_rank": rank_of(exp_net, others),
            "best_team": best[0],
            "best_net": best[1][g],
            "n_teams": len(nets),
            "replayed_now": do_replay,
        }
        path.write_text(json.dumps(row, indent=1, default=str))
        summary["games"][g] = {k: v for k, v in row.items() if k not in ("replay", "score")}
        tag = "" if do_replay else "  (stored estimate, re-priced)"
        print(f"{g:>4} {len(d['items']):>5} | {actual:10.0f} {row['actual_rank']:>4} | {row['pess_net']:11.0f} {row['pess_rank']:>4} | "
              f"{row['exp_net']:10.0f} {row['exp_rank']:>4} | {row['opt_net']:10.0f} {row['opt_rank']:>4} | {best[0]} ({best[1][g]:.0f}){tag}")
    t = verdict(summary["games"], old)
    summary["totals"] = t
    n = t["n_games"]
    print(f"\ntotal: actual {t['actual']:.0f} | replay pess {t['pessimistic']:.0f} | "
          f"exp {t['expected']:.0f} | opt {t['optimistic']:.0f} over {t['n_scored']}/{n} old games")
    if t["missing"]:
        print(f"MISSING replays for old games {t['missing']} - counted as a failure; run: make replay G=\"{' '.join(map(str, t['missing']))}\"")
    print(f"VERDICT: {'SUCCESS' if t['success'] else 'NOT GOOD ENOUGH'} - profitable in "
          f"{t['profitable']}/{n} old games (need > {n // 2}), expected net {t['expected']:.0f} "
          f"[rank, not gating: top-{RANK_TARGET} {t['top_ranked']}/{n}, outright wins {t['wins_exp']}]")
    cal = ROOT / "runs" / "calibration.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        summary["calibration"] = c
        print("calibration in effect:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items() if not isinstance(v, list)})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"saved {OUT.relative_to(ROOT)}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
