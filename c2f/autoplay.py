"""Play every upcoming round automatically:  pixi run python -m c2f.autoplay [--strategy v2] [--lead 90]

  make autoplay            # same thing, v2, launches each round 90 s before it opens

Loop: read the schedule (/api/games/list), pick the next game that has not been played from
this machine (no runs/game_NN.json), sleep until `lead` seconds before its start_time, then
run `make NN` with C2F_STRATEGY set - exactly the command a human would type, so the run log,
truth inference and calibration push all happen as usual. get_case.sh polls for the key, so
launching early costs nothing. After the round's post-processing finishes, go back to the top.

Why a loop over `make NN` rather than something cleverer: rounds 31 and 34 were lost to nobody
pressing the button in time (-69.6k), far more than any pricing change was worth. Presence is
the whole job here; the pricing engine is c2f.run's business.

Stop it with Ctrl-C. It never runs two rounds at once and never re-plays a round that has a log.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time

from c2f.submit import ROOT, list_games

ROUND_S = 60.0  # the window after start_time during which a submission counts


def _ts(s: str) -> float:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def next_game(lead: float) -> tuple[int, float] | None:
    """(game_id, start_ts) of the earliest round still worth playing, else None."""
    now = time.time()
    cands = []
    for g in list_games():
        gid, start = int(g["id"]), _ts(g["start_time"])
        if start + ROUND_S - 5 <= now:          # already over
            continue
        if (ROOT / "runs" / f"game_{gid:02d}.json").exists():
            continue                             # played from this machine
        cands.append((start, gid))
    if not cands:
        return None
    start, gid = min(cands)
    return gid, start


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=os.environ.get("C2F_STRATEGY") or "v2")
    ap.add_argument("--lead", type=float, default=90.0, help="seconds before start_time to launch make NN")
    ap.add_argument("--once", action="store_true", help="play the next round, then exit")
    ap.add_argument("--dry", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args(argv)
    env = {**os.environ, "C2F_STRATEGY": args.strategy}
    while True:
        try:
            nxt = next_game(args.lead)
        except Exception as e:  # noqa: BLE001 - schedule fetch blips must not kill the loop
            print(f"[autoplay] schedule fetch failed ({e}); retrying in 20 s", flush=True)
            time.sleep(20)
            continue
        if nxt is None:
            print("[autoplay] no upcoming rounds on the schedule; checking again in 60 s", flush=True)
            time.sleep(60)
            continue
        gid, start = nxt
        wait = start - args.lead - time.time()
        when = dt.datetime.fromtimestamp(start).strftime("%H:%M:%S")
        print(f"[autoplay] next: game {gid} opens {when} local; launching in {max(wait, 0):.0f} s with strategy {args.strategy}", flush=True)
        if args.dry:
            return 0
        while wait > 0:
            time.sleep(min(wait, 30))
            wait = start - args.lead - time.time()
        print(f"[autoplay] === make {gid} ===", flush=True)
        rc = subprocess.call(["make", str(gid)], cwd=ROOT, env=env)
        print(f"[autoplay] make {gid} exited {rc}", flush=True)
        if args.once:
            return rc


if __name__ == "__main__":
    sys.exit(main())
