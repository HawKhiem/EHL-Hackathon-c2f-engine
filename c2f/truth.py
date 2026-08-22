"""Recover bounds on the secret fair value t per item from the public leaderboard.

  pixi run python -m c2f.truth 2          # game 2 -> runs/truth_game_02.json + table

How: the leaderboard's /matchup cell for (X, Y) in a game is  net_X_vs_Y - net_Y_vs_X  (antisymmetric),
and /transactions tells us every charge a and whether it was accepted. Each pair's cell is a known
piecewise-constant function of the t's, so we keep, per item, only the t-intervals consistent with
every pair equation. Rejections are what carry information: a rejected fair charge contributes 2.5a,
a rejected fraud 0. Accepted charges contribute 2a either way (cap assumed not binding).

Output per item: t_lo (largest charge proven fair) and t_hi (smallest charge proven fraud, or inf).
"""

from __future__ import annotations

import collections
import itertools
import json
import sys

import requests

from c2f.feedback import B, teams, transactions
from c2f.submit import ROOT

TOL = 0.06
MAX_COMBOS = 300_000


def _diff(a: float, acc: bool, t: float) -> float:
    """Contribution of one transaction to (issuer_net - reviewer_net) given t."""
    if a <= 0:
        return 0.0
    if a <= t + 1e-9:
        return 2 * a if acc else 2.5 * a
    return 2 * a if acc else 0.0


def infer(game_id: int) -> dict[int, dict]:
    names = teams()
    tx: dict[tuple, dict] = {}
    for t in names:
        for x in transactions(game_id, t):
            tx[(x["issuer"], x["reviewer"], x["line_item_index"])] = x
    tx_list = list(tx.values())
    games = [g["id"] for g in requests.get(f"{B}/games?completed_only=true&page_size=200", timeout=15).json()["items"]]
    gi = games.index(game_id)
    net = {}
    for t in names:
        for m in requests.get(f"{B}/matchup", params={"team": t}, timeout=15).json()["items"]:
            net[(t, m["opponent"])] = m["cells"][gi]

    items = sorted({x["line_item_index"] for x in tx_list})
    charges = {i: sorted({x["amount"] for x in tx_list if x["line_item_index"] == i and x["amount"] > 0}) for i in items}
    reps = {i: [0.0] + charges[i] for i in items}  # interval k: t in [reps[k], reps[k+1])
    feas = {i: set(range(len(reps[i]))) for i in items}
    by = collections.defaultdict(list)
    for x in tx_list:
        by[(x["issuer"], x["reviewer"])].append(x)

    def contrib(X: str, Y: str, i: int, k: int) -> float:
        t = reps[i][k]
        s = sum(_diff(x["amount"], x["accepted"], t) for x in by[(X, Y)] if x["line_item_index"] == i)
        s -= sum(_diff(x["amount"], x["accepted"], t) for x in by[(Y, X)] if x["line_item_index"] == i)
        return s

    eqs = []
    for (X, Y), v in net.items():
        if X >= Y:
            continue
        inv = [i for i in items if any(x["amount"] > 0 for x in by[(X, Y)] + by[(Y, X)] if x["line_item_index"] == i)]
        if inv:
            eqs.append((X, Y, v, inv))

    changed, rounds = True, 0
    while changed and rounds < 30:
        changed, rounds = False, rounds + 1
        for X, Y, v, inv in eqs:
            prod = 1
            for i in inv:
                prod *= len(feas[i])
            if prod > MAX_COMBOS:
                continue
            tabs = [{k: contrib(X, Y, i, k) for k in feas[i]} for i in inv]
            ok = {i: set() for i in inv}
            for combo in itertools.product(*[sorted(feas[i]) for i in inv]):
                if abs(sum(tabs[j][k] for j, k in enumerate(combo)) - v) <= TOL:
                    for j, k in enumerate(combo):
                        ok[inv[j]].add(k)
            if all(not ok[i] for i in inv):
                continue  # inconsistent (cap?) -> ignore this equation
            for i in inv:
                if ok[i] and ok[i] < feas[i]:
                    feas[i] = ok[i]
                    changed = True

    out = {}
    for i in items:
        ks = sorted(feas[i])
        lo = reps[i][ks[0]]
        hi = reps[i][ks[-1] + 1] if ks[-1] + 1 < len(reps[i]) else None
        out[i] = {"t_lo": lo, "t_hi": hi, "charges": charges[i]}
    return out


def main(game_id: int) -> None:
    res = infer(game_id)
    print(f"game {game_id}: fair-value bounds (t_lo = proven fair, t_hi = proven fraud)")
    for i, v in res.items():
        hi = "inf" if v["t_hi"] is None else f"{v['t_hi']:.0f}"
        print(f"  item {i:2d}: {v['t_lo']:7.2f} <= t < {hi:>5s}   charges seen: {[round(c) for c in v['charges']]}")
    p = ROOT / "runs" / f"truth_game_{game_id:02d}.json"
    p.write_text(json.dumps(res, indent=1))
    print(f"saved {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
