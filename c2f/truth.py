"""Recover bounds on the secret fair value t per item from the public leaderboard, for every
game that has closed. Writes runs/truth_game_NN.json (same shape the history builder reads).

  pixi run python -m c2f.truth          # all completed games missing a truth file
  pixi run python -m c2f.truth 32       # just game 32 (overwrites)

How: the leaderboard /transactions feed lists every (issuer, reviewer, item) with `accepted` and
the PAYOUT `amount` (accepted -> min(a, c); rejected -> a if fair else 0). That gives direct
evidence per item:
  - a rejected charge that was still paid (amount > 0) is proven fair   -> t >= amount
  - a rejected charge paid 0, whose size a is known because some other reviewer accepted it,
    is proven fraud                                                     -> t <  a
t_lo = largest proven-fair charge, t_hi = smallest proven-fraud charge (None if none).

The old engine additionally solved the leaderboard matchup equations to narrow these further;
that only ever tightens the bracket and costs a lot of complexity, so it's deliberately left out.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
FEED_CACHE = RUNS_DIR / "feed"
B = "https://c2f.public.quantco.cloud/leaderboard/api"
_TIMEOUT_S = 15


def truth_path(game_id: int) -> pathlib.Path:
    return RUNS_DIR / f"truth_game_{game_id:02d}.json"


def teams() -> list[str]:
    r = requests.get(f"{B}/matrix", params={"page": 1, "game_limit": 1}, timeout=_TIMEOUT_S)
    r.raise_for_status()
    return [t["team_name"] for t in r.json()["items"]]


def completed_games() -> list[int]:
    r = requests.get(f"{B}/games", params={"completed_only": True, "page_size": 1000}, timeout=_TIMEOUT_S)
    r.raise_for_status()
    return sorted(int(g["id"]) for g in r.json()["items"])


def transactions(game_id: int, team: str) -> list[dict]:
    """Transactions involving `team` in a CLOSED game, cached on disk - a closed game's feed
    never changes, and we pull it once per team."""
    path = FEED_CACHE / f"g{game_id:02d}_{team.replace('/', '_')}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    r = requests.get(
        f"{B}/transactions",
        params={"game_id": game_id, "team": team, "page_size": 1000},
        timeout=_TIMEOUT_S,
    )
    r.raise_for_status()
    items = r.json()["items"]
    FEED_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items), encoding="utf-8")
    return items


def payout_bounds(tx_list: list[dict]) -> dict[int, dict]:
    acc: dict[tuple, float] = collections.defaultdict(float)
    rej: dict[tuple, list[float]] = collections.defaultdict(list)
    for x in tx_list:
        key = (x["issuer"], x["line_item_index"])
        if x["accepted"]:
            acc[key] = max(acc[key], float(x["amount"]))
        else:
            rej[key].append(float(x["amount"]))
    items = sorted({x["line_item_index"] for x in tx_list})
    lo = {i: 0.0 for i in items}
    hi: dict[int, float | None] = {i: None for i in items}
    for (iss, i), amounts in rej.items():
        if max(amounts) > 0:
            lo[i] = max(lo[i], max(amounts))
        elif acc[(iss, i)] > 0:
            a = acc[(iss, i)]
            hi[i] = a if hi[i] is None else min(hi[i], a)
    charges = {
        i: sorted({float(x["amount"]) for x in tx_list if x["line_item_index"] == i and x["amount"] > 0})
        for i in items
    }
    return {i: {"t_lo": lo[i], "t_hi": hi[i], "charges": charges[i]} for i in items}


def infer(game_id: int) -> dict[int, dict]:
    seen: dict[tuple, dict] = {}
    for team in teams():
        for x in transactions(game_id, team):
            seen[(x["issuer"], x["reviewer"], x["line_item_index"])] = x
    return payout_bounds(list(seen.values()))


def write(game_id: int) -> pathlib.Path:
    res = infer(game_id)
    p = truth_path(game_id)
    p.write_text(json.dumps({str(i): v for i, v in res.items()}, indent=1), encoding="utf-8")
    return p


def update() -> list[int]:
    """Write a truth file for every completed game that doesn't have one yet. Returns the
    game ids that were added."""
    added = []
    for game_id in completed_games():
        if truth_path(game_id).exists():
            continue
        write(game_id)
        added.append(game_id)
    return added


if __name__ == "__main__":
    if len(sys.argv) > 1:
        gid = int(sys.argv[1])
        print(f"wrote {write(gid)}")
    else:
        added = update()
        print(f"truth: added {len(added)} game(s): {added}" if added else "truth: up to date")
