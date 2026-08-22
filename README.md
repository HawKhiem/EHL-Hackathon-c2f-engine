# Claim to Fame

QuantCo's *Claim to Fame* hackathon challenge: price and accept insurance claims against
other teams under a 60-second-per-game clock. The brief is in
[GAME_DESCRIPTION.md](GAME_DESCRIPTION.md), the API in [API_HANDBOOK.md](API_HANDBOOK.md).
The solution lives in `c2f/`; its architecture, pipeline and pricing rule are in
[docs/C2F-ARCHITECTURE.md](docs/C2F-ARCHITECTURE.md) — read that before changing `c2f/`,
`tests/`, or the `Makefile`.

## Setup

```bash
curl -fsSL https://pixi.sh/install.sh | sh   # pixi manages Python + deps, no separate install
pixi install
cp .env.example .env                         # fill in TEAM_API_KEY and ANTHROPIC_API_KEY/OPENAI_API_KEY
```

## Playing a game

```bash
make 7            # play game 7: wait for key -> decrypt -> extract -> model -> price -> submit
make check         # real model against the permanent test game 0
make test          # unit tests
```

Start `make N` a few seconds before the game opens — `get_case.sh` polls the key endpoint
every 0.3s and the 60-second clock starts when the key appears. `make N` also commits and
pushes the run log (`runs/game_NN.json`) and, once the game closes, infers fair-value
bounds and refits the calibration (see the Makefile header for the full list of targets:
`play`, `learn`, `fb`, `truth`, `backtest`, `replay`).

## Evaluating a strategy change

```bash
make backtest      # re-score the last 5 completed games with the current price.py + calibration
make replay         # call the model again for every old game (prompt/extract changes)
```

`make backtest` is the single evaluation entry point — see
[docs/C2F-ARCHITECTURE.md](docs/C2F-ARCHITECTURE.md#backtest-gate-for-algorithm-changes) and
`c2f/README.md`.

## Layout

```
c2f/               the engine — extract, digest, LLM call, pricing, submit, backtest
tests/             pytest
cases/             decrypted + zipped per-game cases
runs/              per-game logs, truth/calibration files (committed after each game)
docs/              architecture, design notes
get_case.sh        fetch a game's decryption key + unzip its case
submit.sh          manual submission helper
AGENTS.md          instructions for coding agents  <- read this
```

## Working with coding agents

- **`AGENTS.md`** — architecture, conventions, and the traps worth knowing. Both Claude Code
  and Codex read it. Keep it current.
- **`docs/AGENT-TOOLING.md`** — optional third-party skills, with vetting notes and one-line
  installs.
