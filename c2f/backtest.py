"""Replay the CURRENT strategy on past games and score it against what the other teams actually did.

  pixi run python -m c2f.backtest              # every completed game whose case is decrypted locally
  pixi run python -m c2f.backtest 2 4 6        # specific games
  pixi run python -m c2f.backtest --no-llm 2   # re-score the last replay without calling the model

Per game:  extract -> ensemble (same code path as a live run) -> price -> simulate every pairing
against the real opponents of that round, using the public leaderboard:
  - each opponent's charge a_j per item and whether it was fair (from payouts),
  - each opponent's accept limit b_j as an interval [b_lo, b_hi) (from what they accepted/rejected),
  - the fair value t as an interval [t_lo, t_hi) (largest charge proven fair, smallest proven fraud).
Where an interval leaves the outcome open we score two scenarios: PESSIMISTIC (t = t_lo, opponents
reject whenever they could) and OPTIMISTIC (t just under t_hi, opponents accept whenever they could).
"Would we win?" = our rank if our replayed net replaced our actual net in that round's standings.
EXPECTED = midpoint of the two scenarios; a change is a SUCCESS only if the expected replay wins
(rank 1) in more than half of the replayed games. The pre-commit hook enforces that.

Writes runs/backtest/game_NN.json and runs/backtest/summary.json (+ a table on stdout).
The pre-commit hook (.githooks/pre-commit) refuses algorithm commits without a fresh summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from c2f import llm
from c2f.ensemble import aggregate
from c2f.extract import load_case
from c2f.feedback import B, DEFAULT_TEAM, digest
from c2f.price import price_all
from c2f.run import N_FULL, merge_estimates
from c2f.submit import ROOT

OUT = ROOT / "runs" / "backtest"
INF = float("inf")


# ----------------------------------------------------------------------------- replay
def replay(game_id: int, n_votes: int = N_FULL) -> dict:
    case = load_case(ROOT / "cases" / f"case_{game_id:02d}", game_id)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_votes) as ex:
        futs = [ex.submit(llm.estimate, case, timeout=60, strict=False) for _ in range(n_votes)]
        outs, errors = [], []
        for f in futs:
            try:
                out, meta = f.result()
                outs.append(out)
            except Exception as e:  # noqa: BLE001
                errors.append(str(e))
    if not outs:
        raise RuntimeError(f"all votes failed: {errors}")
    agg = aggregate(outs)
    est = merge_estimates(case, agg)
    rows = price_all(est)
    return {"rows": rows, "ensemble": agg, "votes": len(outs), "errors": errors, "seconds": round(time.time() - t0, 1)}


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


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*", type=int)
    ap.add_argument("--no-llm", action="store_true", help="re-score the stored replay, don't call the model")
    ap.add_argument("--votes", type=int, default=N_FULL)
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    gids, nets = actual_nets()
    us = DEFAULT_TEAM
    games = args.games or [g for g in gids if (ROOT / "cases" / f"case_{g:02d}" / "policy.txt").exists()]
    summary = {"team": us, "games": {}, "generated_at": time.time()}
    print(f"backtest as {us} on games {games}\n")
    print(f"{'game':>4} {'items':>5} | {'actual net':>10} {'rank':>4} | {'replay pess':>11} {'rank':>4} | {'replay exp':>10} {'rank':>4} | {'replay opt':>10} {'rank':>4} | best team")
    for g in games:
        if g not in gids:
            print(f"{g:>4}  not completed yet, skipped")
            continue
        case_dir = ROOT / "cases" / f"case_{g:02d}"
        if not (case_dir / "policy.txt").exists():
            print(f"{g:>4}  case not decrypted locally ({case_dir}), skipped")
            continue
        path = OUT / f"game_{g:02d}.json"
        if args.no_llm:
            if not path.exists():
                print(f"{g:>4}  no stored replay, run without --no-llm")
                continue
            rep = json.loads(path.read_text())["replay"]
        else:
            rep = replay(g, args.votes)
        d = digest(g)
        sc = score(g, rep["rows"], d, us)
        others = [nets[t][g] for t in nets if t != us]
        best = max(nets.items(), key=lambda kv: kv[1][g])
        actual = nets.get(us, {}).get(g, float("nan"))
        row = {
            "actual_net": actual,
            "actual_rank": rank_of(actual, others),
            "replay": rep,
            "score": sc,
            "pess_net": sc["scenarios"]["pessimistic"]["net"],
            "pess_rank": rank_of(sc["scenarios"]["pessimistic"]["net"], others),
            "opt_net": sc["scenarios"]["optimistic"]["net"],
            "opt_rank": rank_of(sc["scenarios"]["optimistic"]["net"], others),
            "exp_net": round((sc["scenarios"]["pessimistic"]["net"] + sc["scenarios"]["optimistic"]["net"]) / 2, 2),
            "exp_rank": rank_of((sc["scenarios"]["pessimistic"]["net"] + sc["scenarios"]["optimistic"]["net"]) / 2, others),
            "best_team": best[0],
            "best_net": best[1][g],
            "n_teams": len(nets),
        }
        path.write_text(json.dumps(row, indent=1, default=str))
        summary["games"][g] = {k: v for k, v in row.items() if k not in ("replay", "score")}
        print(f"{g:>4} {len(d['items']):>5} | {actual:10.0f} {row['actual_rank']:>4} | {row['pess_net']:11.0f} {row['pess_rank']:>4} | "
              f"{row['exp_net']:10.0f} {row['exp_rank']:>4} | {row['opt_net']:10.0f} {row['opt_rank']:>4} | {best[0]} ({best[1][g]:.0f})")
    if summary["games"]:
        tot_a = sum(v["actual_net"] for v in summary["games"].values())
        tot_p = sum(v["pess_net"] for v in summary["games"].values())
        tot_o = sum(v["opt_net"] for v in summary["games"].values())
        tot_e = sum(v["exp_net"] for v in summary["games"].values())
        wins_p = sum(1 for v in summary["games"].values() if v["pess_rank"] == 1)
        wins_e = sum(1 for v in summary["games"].values() if v["exp_rank"] == 1)
        wins_o = sum(1 for v in summary["games"].values() if v["opt_rank"] == 1)
        n = len(summary["games"])
        success = wins_e * 2 > n
        summary["totals"] = {"actual": tot_a, "pessimistic": tot_p, "expected": tot_e, "optimistic": tot_o,
                             "wins_pess": wins_p, "wins_exp": wins_e, "wins_opt": wins_o, "n_games": n, "success": success}
        print(f"\ntotal: actual {tot_a:.0f} | replay pess {tot_p:.0f} ({wins_p} wins) | exp {tot_e:.0f} ({wins_e} wins) | opt {tot_o:.0f} ({wins_o} wins) over {n} games")
        print(f"VERDICT: {'SUCCESS' if success else 'NOT GOOD ENOUGH'} - expected replay wins {wins_e}/{n} old games (need > {n // 2})")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"saved {OUT.relative_to(ROOT)}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
