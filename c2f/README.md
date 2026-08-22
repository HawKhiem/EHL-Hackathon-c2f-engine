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
