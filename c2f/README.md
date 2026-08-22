# c2f — Claim to Fame engine

One command per game, must finish inside the 60 s window:

    pixi run python -m c2f.run 7          # play game 7 (polls key, decrypts, prices, submits)
    pixi run python -m c2f.run 0 --no-submit --case-dir cases/case_00   # offline dry run
    pixi run python -m pytest -q tests

Pipeline: `get_case.sh` → `extract.py` (case dict) → `llm.py` (one model call, policy and
description verbatim) → `price.py` (a, b) → `submit.py` (PUT) + `runs/game_NN.json`.

Two model passes run in parallel: a fast one is submitted as soon as it lands, the full one
overwrites it if it arrives before ~56 s. Keys: set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
in `.env` (see `.env.example`). Design: `docs/superpowers/specs/2026-08-22-c2f-engine-design.md`.

## Backtest — required before changing the algorithm

`make backtest` replays the current strategy on every past game whose case is decrypted
locally and scores it against what the other teams actually did that round (their charges,
their accept/reject limits and the inferred fair-value bounds from the public leaderboard).
It prints, per game, our actual net and rank vs the replayed net and rank (pessimistic /
optimistic where the data leaves an outcome open) and writes `runs/backtest/summary.json`.

    make hooks                 # once per clone: installs the pre-commit hook
    make backtest              # all completed games (calls the model, ~40 s per game)
    make backtest G="2 4 6"    # specific games
    make rescore               # re-score stored replays without the model (price.py-only changes)

**Success criterion: the replayed strategy must win (rank 1 on the expected net) in more than
half of the old games.** The hook refuses a commit that touches
`c2f/{price,llm,ensemble,extract,run,calibrate}.py` unless `runs/backtest/summary.json` is newer
than the changed files, staged with them, and says `success: true`. Knowing override for
emergencies: `ALLOW_BACKTEST_FAIL=1 git commit ...` (say why in the message).
So the workflow is: edit -> `make backtest` (or `make rescore`) -> look at the table ->
`git add runs/backtest` -> commit.
