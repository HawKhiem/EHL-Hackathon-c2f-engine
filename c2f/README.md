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

`make backtest` is the **single evaluation entry point** — it folds in the other analysis
tools as components: `c2f.feedback` (opponents' charges / accept-reject / payout-based `t`
bounds), `c2f.truth` (tighter `t` bounds from the matchup cells, cached in
`runs/truth_game_NN.json`) and the calibration in effect (`runs/calibration.json`, reported in
the summary). `make fb` / `make truth` / `make learn` remain as building blocks for looking at
one game or re-fitting the calibration; a strategy is judged only by `make backtest`.

It replays the current strategy on every past game whose case is decrypted
locally and scores it against what the other teams actually did that round (their charges,
their accept/reject limits and the inferred fair-value bounds from the public leaderboard).
It prints, per game, our actual net and rank vs the replayed net and rank (pessimistic /
optimistic where the data leaves an outcome open) and writes `runs/backtest/summary.json`.

    make backtest              # re-price + re-score stored estimates with the current price.py + calibration (no model call)
    make replay                # call the model again for every old game (prompt / extract / policy changes)
    make replay G="2 4 6"      # call the model for these games only

**Success criterion: the replayed strategy must MAKE MONEY, consistently.** The verdict is
SUCCESS when the expected replay net is positive in more than half the old games and the total
expected net is positive. Rank is printed (top-3 = `RANK_TARGET`, outright wins) but does **not**
gate: a steady 3rd place every round while in profit is the target outcome, not a failure.
Nothing enforces this — a pre-commit hook that blocked commits to
`c2f/{price,llm,policy,extract,run,calibrate}.py` without a fresh passing summary was removed
as too much friction. Run it when a change is worth checking: edit -> `make backtest` (or
`make replay` if the change is upstream of pricing) -> look at the table.
