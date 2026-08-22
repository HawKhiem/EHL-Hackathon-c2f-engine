"""After a game closes: what did everyone charge, who accepted what, how did we do.

  pixi run python -m c2f.feedback 2            # digest game 2
  C2F_TEAM="our name"  (optional; otherwise matched from runs/game_NN.json)

Per item it prints every team's charge a (their t estimate!), how many reviewers
accepted it, and the bounds on each team's acceptance limit b implied by their
accept/reject decisions. Market median of a across teams is a cheap second opinion on t.
"""

from __future__ import annotations

import collections
import json
import os
import sys
from statistics import median

import requests

from c2f.submit import ROOT

B = "https://c2f.public.quantco.cloud/leaderboard/api"


def teams() -> list[str]:
    return [t["team_name"] for t in requests.get(f"{B}/matrix?page=1&game_limit=1", timeout=15).json()["items"]]


def transactions(game_id: int, team: str) -> list[dict]:
    r = requests.get(f"{B}/transactions", params={"game_id": game_id, "team": team, "page_size": 1000}, timeout=15)
    r.raise_for_status()
    return r.json()["items"]


def digest(game_id: int) -> dict:
    names = teams()
    issued: dict[str, dict[int, float]] = collections.defaultdict(dict)
    accepted: dict[str, dict[int, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    b_lo: dict[str, dict[int, float]] = collections.defaultdict(dict)  # reviewer accepted a -> b >= a
    b_hi: dict[str, dict[int, float]] = collections.defaultdict(dict)  # reviewer rejected a -> b < a
    seen = set()
    for t in names:
        for x in transactions(game_id, t):
            key = (x["issuer"], x["reviewer"], x["line_item_index"])
            if key in seen:
                continue
            seen.add(key)
            i, a = x["line_item_index"], x["amount"]
            issued[x["issuer"]][i] = a
            accepted[x["issuer"]][i].append(x["accepted"])
            rv = x["reviewer"]
            if x["accepted"]:
                b_lo[rv][i] = max(b_lo[rv].get(i, 0.0), a)
            elif a > 0:
                b_hi[rv][i] = min(b_hi[rv].get(i, float("inf")), a)
    items = sorted({i for t in issued for i in issued[t]})
    return {"teams": names, "items": items, "issued": issued, "accepted": accepted, "b_lo": b_lo, "b_hi": b_hi}


def our_team(game_id: int, d: dict) -> str | None:
    if os.environ.get("C2F_TEAM"):
        return os.environ["C2F_TEAM"]
    p = ROOT / "runs" / f"game_{game_id:02d}.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text())
    subs = [s for s in rec.get("submissions", []) if isinstance(s.get("response"), list)]
    if not subs:
        return None
    ours = {r["index"]: r["charge_price"] for r in subs[-1]["rows"]}
    best, best_n = None, -1
    for t in d["teams"]:
        n = sum(1 for i, a in ours.items() if abs(d["issued"][t].get(i, -1) - a) < 0.01)
        if n > best_n:
            best, best_n = t, n
    return best if best_n >= max(1, len(ours) // 2) else None


def main(game_id: int) -> None:
    d = digest(game_id)
    us = our_team(game_id, d)
    print(f"game {game_id}: {len(d['teams'])} teams, items {d['items']}   us = {us or '?'}")
    for i in d["items"]:
        charges = [(d["issued"][t].get(i, 0.0), t) for t in d["teams"]]
        nonzero = [a for a, _ in charges if a > 0]
        med = median(nonzero) if nonzero else 0.0
        print(f"\n# item {i}: market median a = {med:.0f}  ({len(nonzero)} teams charged > 0)")
        for a, t in sorted(charges, reverse=True):
            acc = d["accepted"][t][i]
            lo, hi = d["b_lo"][t].get(i), d["b_hi"][t].get(i)
            b_txt = f"b in [{lo if lo is not None else 0:.0f}, {hi if hi is not None else float('inf'):.0f})"
            mark = " <== us" if t == us else ""
            print(f"   {t:22s} a={a:8.1f}  accepted {sum(acc):2d}/{len(acc):2d}   {b_txt}{mark}")
    if us:
        perf = requests.get(f"{B}/performance", params={"team": us}, timeout=15).json()
        print("\nour performance so far:", json.dumps(perf))
    out = ROOT / "runs" / f"feedback_game_{game_id:02d}.json"
    out.write_text(json.dumps({k: v for k, v in d.items()}, indent=1, default=str))
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
