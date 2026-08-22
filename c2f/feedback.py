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
DEFAULT_TEAM = "Nullpointer Naan"  # identified from game-2 bounds; override with C2F_TEAM


def teams() -> list[str]:
    return [t["team_name"] for t in requests.get(f"{B}/matrix?page=1&game_limit=1", timeout=15).json()["items"]]


def transactions(game_id: int, team: str) -> list[dict]:
    r = requests.get(f"{B}/transactions", params={"game_id": game_id, "team": team, "page_size": 1000}, timeout=15)
    r.raise_for_status()
    return r.json()["items"]


def digest(game_id: int) -> dict:
    """amount in the feed is the PAYOUT: accepted -> min(a, c); rejected -> a if fair (a <= t) else 0."""
    names = teams()
    pay: dict[str, dict[int, list]] = collections.defaultdict(lambda: collections.defaultdict(list))  # issuer -> item -> [(reviewer, accepted, payout)]
    seen = set()
    for t in names:
        for x in transactions(game_id, t):
            key = (x["issuer"], x["reviewer"], x["line_item_index"])
            if key in seen:
                continue
            seen.add(key)
            pay[x["issuer"]][x["line_item_index"]].append((x["reviewer"], bool(x["accepted"]), float(x["amount"])))
    items = sorted({i for t in pay for i in pay[t]})
    issued: dict[str, dict[int, float | None]] = collections.defaultdict(dict)  # best guess of a
    fair: dict[str, dict[int, bool | None]] = collections.defaultdict(dict)    # a <= t ?
    t_lo: dict[int, float] = {i: 0.0 for i in items}
    t_hi: dict[int, float] = {i: float("inf") for i in items}
    for iss in names:
        for i in items:
            rows = pay[iss].get(i, [])
            acc_pay = [p for _, a, p in rows if a]
            rej_pay = [p for _, a, p in rows if not a]
            a_guess = max(acc_pay) if acc_pay else (max(rej_pay) if rej_pay and max(rej_pay) > 0 else None)
            issued[iss][i] = a_guess
            if rej_pay:
                if max(rej_pay) > 0:
                    fair[iss][i] = True
                    t_lo[i] = max(t_lo[i], max(rej_pay))
                elif a_guess and a_guess > 0:
                    fair[iss][i] = False
                    t_hi[i] = min(t_hi[i], a_guess)
                elif not acc_pay:
                    fair[iss][i] = None  # rejected everywhere with 0: fraud, but a unknown (>0) or a=0
            else:
                fair[iss][i] = None
    # reviewer limits: accepted a -> b >= a ; rejected a -> b < a
    b_lo: dict[str, dict[int, float]] = collections.defaultdict(dict)
    b_hi: dict[str, dict[int, float]] = collections.defaultdict(dict)
    for iss in names:
        for i in items:
            a = issued[iss][i]
            for rv, acc, _ in pay[iss].get(i, []):
                if a is None:
                    continue
                if acc:
                    b_lo[rv][i] = max(b_lo[rv].get(i, 0.0), a)
                elif a > 0:
                    b_hi[rv][i] = min(b_hi[rv].get(i, float("inf")), a)
    accepted = {iss: {i: [a for _, a, _ in pay[iss].get(i, [])] for i in items} for iss in names}
    return {"teams": names, "items": items, "issued": issued, "fair": fair, "t_lo": t_lo, "t_hi": t_hi,
            "accepted": accepted, "b_lo": b_lo, "b_hi": b_hi, "pay": pay}


def our_team(game_id: int, d: dict) -> str | None:
    if os.environ.get("C2F_TEAM"):
        return os.environ["C2F_TEAM"]
    return DEFAULT_TEAM
    p = ROOT / "runs" / f"game_{game_id:02d}.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text())
    subs = [s for s in rec.get("submissions", []) if isinstance(s.get("response"), list)]
    if not subs:
        return None
    ours_a = {r["index"]: r["charge_price"] for r in subs[-1]["rows"]}
    ours_b = {r["index"]: r["acceptance_limit"] for r in subs[-1]["rows"]}
    best, best_n = None, -1
    for t in d["teams"]:
        n = 0
        for i in ours_a:
            a = d["issued"][t].get(i)
            if a is not None and abs(a - ours_a[i]) < 0.01:
                n += 2
            lo, hi = d["b_lo"][t].get(i, 0.0), d["b_hi"][t].get(i, float("inf"))
            if lo <= ours_b[i] < hi:
                n += 1
        if n > best_n:
            best, best_n = t, n
    return best if best_n >= len(ours_a) else None


def main(game_id: int) -> None:
    d = digest(game_id)
    us = our_team(game_id, d)
    print(f"game {game_id}: {len(d['teams'])} teams, items {d['items']}   us = {us or '?'}")
    ours = None
    p = ROOT / "runs" / f"game_{game_id:02d}.json"
    if p.exists():
        rec = json.loads(p.read_text())
        subs = [s for s in rec.get("submissions", []) if isinstance(s.get("response"), list)]
        if subs:
            ours = {r["index"]: r for r in subs[-1]["rows"]}
    print(f"{'item':>4} {'t_lo':>7} {'t_hi':>7} {'mkt_med':>7} | {'our a':>7} {'our b':>7}  verdict")
    for i in d["items"]:
        charges = [a for t in d["teams"] for a in [d["issued"][t].get(i)] if a]
        med = median(charges) if charges else 0.0
        lo, hi = d["t_lo"][i], d["t_hi"][i]
        if ours and i in ours:
            a, b = ours[i]["charge_price"], ours[i]["acceptance_limit"]
            v = []
            if a > 0 and hi < float("inf") and a >= hi:
                v.append("a TOO HIGH (fraud)")
            if a > 0 and a <= lo and lo > 0:
                v.append("a fair")
            if b < lo:
                v.append("b TOO LOW (penalties)")
            if hi < float("inf") and b >= hi:
                v.append("b too high (paid fraud)")
            a_s, b_s, v_s = f"{a:7.1f}", f"{b:7.1f}", ", ".join(v) or "-"
        else:
            a_s = b_s = "      ?"
            v_s = "-"
        print(f"{i:>4} {lo:7.0f} {hi if hi < float('inf') else 0:7.0f} {med:7.0f} | {a_s} {b_s}  {v_s}")
    print("\n(t_hi 0 = unknown)  per item, charges by team (a, fair?, accepted):")
    for i in d["items"]:
        cells = []
        for t in d["teams"]:
            a = d["issued"][t].get(i)
            if a:
                f = d["fair"][t].get(i)
                acc = d["accepted"][t][i]
                cells.append(f"{t[:12]}={a:.0f}{'✓' if f else ('✗' if f is False else '?')}({sum(acc)}/{len(acc)})")
        print(f"  #{i}: " + "  ".join(cells))
    if us:
        perf = requests.get(f"{B}/performance", params={"team": us}, timeout=15).json()
        print("\nour performance so far:", json.dumps(perf))
    out = ROOT / "runs" / f"feedback_game_{game_id:02d}.json"
    slim = {k: v for k, v in d.items() if k != "pay"}
    out.write_text(json.dumps(slim, indent=1, default=str))
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
