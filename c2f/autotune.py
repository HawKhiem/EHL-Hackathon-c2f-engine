"""Turn the post-mortem's causes into constant changes, but only ones that survive a gate.

  pixi run python -m c2f.autotune             # propose only, change nothing
  pixi run python -m c2f.autotune --apply     # write runs/tuning.json

Flow: c2f.truth -> c2f.calibrate -> c2f.postmortem -> here. The post-mortem says
which mistake cost what; this asks whether any tunable constant would have fixed
it, re-scores every stored round with the change, and accepts it only if it
passes both halves of the gate.

The gate, and why it is the whole point
---------------------------------------
A change is accepted only when it improves the total AND improves a strict
majority of individual games. Total alone is not enough, and this project has
three separate proofs of that:

* `B_QUANTILE` swept over eight games showed an "optimum" at 0.15, 3,655 better
  in total. Per game it was cheaper in three and dearer in four - the entire gain
  came from game 9 alone. Applying it would have been fitting one round.
* The accept-limit ratio fitted at 0.66 on six games and 0.44 on eight, while
  per-game signs stayed mixed. Two fits, two answers, no signal.
* `K_UNCERTAINTY` was moved 0.25 -> 0.5 on one round's evidence, then back.

So the sign test is not a refinement, it is the thing that stops the loop
oscillating. A proposal that fails it is logged with its numbers rather than
silently dropped, so the next person can see what was already tried.

What this deliberately will NOT do
----------------------------------
Edit prompts. A prompt rewrite driven by one round's failures is how the two
sides start fighting: an offline A/B that widened the estimate band to make `b`
safer also shaved `a` through the spread term and lost more income than it saved.
Prompt changes need a human reading the actual failures.

It also cannot fix the largest causes. ABSTENTION, COVERAGE_MISS, NO_SUBMISSION
and MISSED_CHARGE are not functions of any constant - a model that declines to
value an item, or calls a covered item junk, produces a=0 and b=0 whatever the
constants say. Those are reported as human work, not tuned around.
"""

from __future__ import annotations

import argparse
import json
import sys

from c2f import backtest, postmortem
from c2f import price as P
from c2f.submit import ROOT

_DIGEST: dict[int, dict] = {}


def digest_once(game_id: int) -> dict:
    """backtest.digest is pure once the feed is cached, but it is not free - the
    candidate loop would otherwise rebuild the same digest for every trial."""
    if game_id not in _DIGEST:
        _DIGEST[game_id] = backtest.digest(game_id)
    return _DIGEST[game_id]

TUNING_PATH = ROOT / "runs" / "tuning.json"

#: constant -> (candidate values, human note). Deliberately a short hand-authored
#: list rather than a search: a fine grid over six constants finds noise.
CANDIDATES: dict[str, tuple[tuple[float, ...], str]] = {
    "RISK_AVERSION": ((0.30, 0.585, 0.85), "penalty on the sd of the per-opponent payout"),
    "B_QUANTILE": ((0.20, 0.27, 0.3333), "accept limit as a quantile of the belief"),
    "UNCOVERED_CHARGE": ((0.40, 0.60, 0.90), "free shot on an item judged worthless"),
}

#: cause -> the constants that could plausibly move it, and in which direction.
CAUSE_LEVERS: dict[str, list[str]] = {
    "UNDERCHARGE": ["RISK_AVERSION"],
    "OVERCHARGE": ["RISK_AVERSION"],
    "UNDER_ESTIMATE": ["B_QUANTILE"],
    "OVER_ESTIMATE": ["B_QUANTILE"],
}

#: causes no constant can address - reported for a human instead
HUMAN_ONLY = {
    "ABSTENTION": "guard price_item: a covered call with no number must never ship b=0",
    "COVERAGE_MISS": "prompt/coverage work - the item was called uncovered and was not",
    "MISSED_CHARGE": "same root as ABSTENTION/COVERAGE_MISS: a=0 on an item worth money",
    "NO_SUBMISSION": "scheduling, not modelling",
    "FAST_ONLY": "the full pass did not land - check its timeout",
}


def scored_games() -> list[int]:
    """Games with a stored replay we can re-price without calling the model."""
    return sorted(
        int(p.stem.split("_")[-1])
        for p in backtest.OUT.glob("game_*.json")
        if (ROOT / "cases" / f"case_{int(p.stem.split('_')[-1]):02d}" / "policy.txt").exists()
    )


def evaluate(games: list[int], us: str, nets: dict) -> dict[int, dict]:
    """Per-game rows in the shape `backtest.verdict` consumes, under the price
    module's CURRENT constants. No model call - stored estimates are re-priced."""
    out: dict[int, dict] = {}
    for g in games:
        stored = json.loads((backtest.OUT / f"game_{g:02d}.json").read_text(encoding="utf-8"))
        rep = backtest.reprice(g, stored["replay"])
        sc = backtest.score(g, rep["rows"], digest_once(g), us)
        pess = sc["scenarios"]["pessimistic"]["net"]
        opt = sc["scenarios"]["optimistic"]["net"]
        exp = (pess + opt) / 2
        others = [nets[t][g] for t in nets if t != us and g in nets[t]]
        out[g] = {
            "actual_net": nets.get(us, {}).get(g, 0.0),
            "pess_net": pess, "pess_rank": backtest.rank_of(pess, others),
            "exp_net": exp, "exp_rank": backtest.rank_of(exp, others),
            "opt_net": opt, "opt_rank": backtest.rank_of(opt, others),
        }
    return out


def trial(name: str, value: float, games: list[int], us: str, nets: dict) -> dict[int, dict]:
    """Re-score every game with one constant temporarily changed."""
    before = getattr(P, name)
    try:
        setattr(P, name, value)
        return evaluate(games, us, nets)
    finally:
        setattr(P, name, before)


def gate(base: dict[int, dict], cand: dict[int, dict], old: list[int]) -> dict:
    """Three conditions, all required.

    1. the total expected net improves;
    2. a strict majority of individual games improve - the sign test that stops
       one outlier round from moving a constant;
    3. `backtest.verdict` still reports SUCCESS over the last WINDOW games. A
       change that improves the total while dropping the strategy out of a
       passing state is not an improvement worth shipping.
    """
    deltas = {g: cand[g]["exp_net"] - base[g]["exp_net"] for g in base}
    wins = sum(1 for d in deltas.values() if d > 1)
    losses = sum(1 for d in deltas.values() if d < -1)
    total = sum(deltas.values())
    v = backtest.verdict(cand, old)
    base_v = backtest.verdict(base, old)
    return {
        "total_delta": total,
        "wins": wins,
        "losses": losses,
        "ties": len(deltas) - wins - losses,
        "per_game": {str(g): round(d, 2) for g, d in deltas.items()},
        "backtest_success": bool(v["success"]),
        "backtest_profitable": f"{v['profitable']}/{v['n_games']}",
        "baseline_success": bool(base_v["success"]),
        "accept": total > 0 and wins > losses and bool(v["success"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write runs/tuning.json")
    ap.add_argument("--team", default="AsianSuperNerds")
    args = ap.parse_args(argv)

    games = scored_games()
    if len(games) < 3:
        print(f"only {len(games)} stored replays - need at least 3 for the sign test")
        return 2

    # ---- 1. what went wrong, and how much of it is even tunable ----
    causes: dict[str, float] = {}
    for g in games:
        try:
            res = postmortem.analyse(g, args.team)
        except FileNotFoundError:
            continue
        for f in res["findings"]:
            causes[f["cause"]] = causes.get(f["cause"], 0.0) + f["euros"]

    tunable = {c: v for c, v in causes.items() if c in CAUSE_LEVERS}
    human = {c: v for c, v in causes.items() if c in HUMAN_ONLY}
    print(f"post-mortem over games {games}\n")
    print(f"  {'euros':>11}  cause            addressable by a constant?")
    for c, v in sorted(causes.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>11,.0f}  {c:<15}  {'yes' if c in CAUSE_LEVERS else 'NO - human work'}")
    if human:
        print(f"\n  {sum(human.values()):,.0f} EUR is NOT reachable by tuning:")
        for c, v in sorted(human.items(), key=lambda kv: -kv[1]):
            print(f"    {v:>11,.0f}  {c:<15} {HUMAN_ONLY[c]}")

    if not tunable:
        print("\nnothing tunable in this window. Do the human work above first.")
        return 0

    levers = sorted({n for c in tunable for n in CAUSE_LEVERS[c]})
    print(f"\ntunable causes point at: {levers}")

    # ---- 2. baseline, then every candidate through the gate ----
    gids, nets = backtest.actual_nets()
    old = backtest.old_games(gids)
    base = evaluate(games, args.team, nets)
    base_v = backtest.verdict(base, old)
    print(f"\nbaseline expected net over {len(games)} games: "
          f"{sum(r['exp_net'] for r in base.values()):,.0f}")
    print(f"baseline backtest verdict over the last {len(old)} {old}: "
          f"{'SUCCESS' if base_v['success'] else 'FAIL'}"
          f" (profitable {base_v['profitable']}/{base_v['n_games']})")
    print(f"\n{'constant':<18}{'value':>8}{'total delta':>13}{'W-L-T':>10}  verdict")
    accepted: dict[str, float] = {}
    audit: list[dict] = []
    for name in levers:
        values, note = CANDIDATES[name]
        if not hasattr(P, name):
            # price.py gets refactored; a lever can vanish (K_UNCERTAINTY did when the
            # charge moved to best_charge). Skip it loudly rather than crash the loop.
            print(f"{name:<18}{'-':>8}{'-':>13}{'-':>10}  skipped: no longer in price.py")
            continue
        current = getattr(P, name)
        best: tuple[float, dict] | None = None
        for v in values:
            if abs(v - current) < 1e-9:
                continue
            g = gate(base, trial(name, v, games, args.team, nets), old)
            audit.append({"constant": name, "value": v, "current": current, **g})
            if g["accept"]:
                flag = "ACCEPT"
            elif not g["backtest_success"]:
                flag = f"rejected: backtest FAILS ({g['backtest_profitable']} profitable)"
            elif g["total_delta"] > 0:
                flag = f"rejected: total up but only {g['wins']}/{g['wins'] + g['losses']} games"
            else:
                flag = "rejected: worse overall"
            record = f"{g['wins']}-{g['losses']}-{g['ties']}"
            print(f"{name:<18}{v:>8.4f}{g['total_delta']:>13,.0f}{record:>10}  {flag}")
            if g["accept"] and (best is None or g["total_delta"] > best[1]["total_delta"]):
                best = (v, g)
        if best:
            accepted[name] = best[0]

    # ---- 3. write, or not ----
    if not accepted:
        print("\nno candidate passed the gate. Constants stay as they are - that is a result,"
              "\nnot a failure: it means the remaining loss is not in these numbers.")
    else:
        print("\naccepted:")
        for name, v in accepted.items():
            print(f"  {name}: {getattr(P, name)} -> {v}")
    payload = {"accepted": accepted, "audit": audit, "causes": causes, "games": games}
    if args.apply:
        TUNING_PATH.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\nwrote {TUNING_PATH.relative_to(ROOT)}"
              + ("" if accepted else " (no overrides - price.py behaviour unchanged)"))
    else:
        print("\nproposal only; pass --apply to write runs/tuning.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
