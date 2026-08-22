# AGENTS.md

Instructions for coding agents (Claude Code, Codex, Cursor) working in this repo.
Humans: read `README.md` first, then this.

---

## What this is

QuantCo's *Claim to Fame* hackathon challenge. The brief is
[GAME_DESCRIPTION.md](GAME_DESCRIPTION.md), the API is
[API_HANDBOOK.md](API_HANDBOOK.md). The solution is `c2f/` — a pipeline that decrypts a
game's case, prices each invoice line item against a secret fair value, and submits within
a 60-second window.

**Read [docs/C2F-ARCHITECTURE.md](docs/C2F-ARCHITECTURE.md) before touching `c2f/`,
`tests/`, or the `Makefile`.** It has the pipeline, the prompt layout, and the pricing rule
— duplicating it here would just go stale.

| Layer | Stack | Lives in |
|---|---|---|
| Engine | Python (pixi-managed), OpenAI | `c2f/` |
| Tests | pytest | `tests/` |
| Per-game logs, truth/calibration | committed JSON | `runs/` |
| Decrypted + zipped cases | | `cases/` |

This team runs on OpenAI only — `ANTHROPIC_API_KEY` stays unset in `.env` so
`c2f/llm.py` picks OpenAI. Do not add Anthropic-only code paths without asking.

---

## Run it

```bash
pixi install                # first time only — provisions Python + deps
make N                       # play game N: wait for key -> decrypt -> extract -> model -> price -> submit
make check                   # real model against the permanent test game 0
make test                    # unit tests
```

Start `make N` a few seconds before the game opens — see `docs/C2F-ARCHITECTURE.md` for the
per-second timeline. `make N` also commits and pushes `runs/game_NN.json`, then infers truth
bounds and refits calibration once the game closes. Other Makefile targets: `play`, `learn`,
`fb`, `truth`, `backtest`, `replay` — see the Makefile header for what each does.

**Python is managed entirely by [`pixi`](https://pixi.sh/).** Never call `pip` or invoke
`python` directly — prefix with `pixi run`:

```bash
pixi run python -m pytest -q tests
pixi add <package>            # add a dep (updates pixi.toml + pixi.lock)
```

Dependencies live in `pixi.toml`. `pixi.lock` is **committed** so everyone resolves
identical versions — commit it whenever it changes.

---

## Where things go

```
c2f/
  extract.py       case dict from cases/case_NN/ (policy, description, invoice, images)
  policy.py        policy digest — the parts that bind, prepended to the verbatim policy
  llm.py           the model call(s) — fast pass + full pass
  price.py         (a, b) per item from the model's t_low/mid/high belief
  run.py           orchestrates: digest + fast pass in parallel, full pass overwrites
  submit.py        PUT to the submissions API, logs to runs/
  feedback.py       opponents' charges/accept-reject, payout-based t bounds for a finished game
  truth.py          tighter t bounds from the matchup cells -> runs/truth_game_NN.json
  calibrate.py      refits bias/sigma/acceptance from every truth + feedback file
  backtest.py       THE evaluation entry point — see below

tests/              pytest, one file per c2f module
get_case.sh         polls the key endpoint, decrypts + unzips a game's case
submit.sh           manual submission helper (INDEX:CHARGE:LIMIT or a JSON file)
```

---

## Evaluating a strategy change

`make backtest` is the single evaluation entry point — it replays the current strategy
against the real opponents of past games (their charges, their implied accept limits, the
inferred `t` bounds) and reports actual vs. replayed net and rank per game.

```bash
make backtest        # re-price + re-score stored estimates, current price.py + calibration — no model call
make replay           # call the model again for every old game (prompt/extract/policy changes)
make replay G="2 4 6" # call the model for these games only
```

**Success criterion: the replayed strategy must win (rank 1 on expected net) in more than
half of the old games**, over the last 5 completed games with a decrypted case (`WINDOW`).
Not enforced by a hook — run it when a change is worth checking. See
`docs/C2F-ARCHITECTURE.md#backtest-gate-for-algorithm-changes` and `c2f/README.md`.

---

## Conventions

- `decision`-shaped code (`price.py`, `backtest.py`) should stay pure and unit-tested —
  it's the only place a silent 10x pricing error survives review.
- Never let the LLM do arithmetic across items; parse quantities in code and cross-check
  against the model's stated total.
- Log everything on the hot path to `runs/`; nothing after the submit POST should be able
  to block a submission.

---

## Writing anything a human will read

Pitch copy, the project description, README text, demo scripts — these are **judged**.
Write them like a person who built the thing:

- Lead with what it does and who it is for. No "In today's fast-paced world."
- Concrete over abstract: name the actual number, the actual before/after.
- Cut hedges ("arguably", "we believe"), cut triples ("fast, scalable, and robust").
- Claim only what the demo actually does. Judges probe. Aspirational claims collapse.

---

## Before you say you are done

```bash
pixi run python -m pytest -q tests
```

CI (`.github/workflows/ci.yml`) runs the same command on every push and PR. Then, before a
live round, actually run `make check` against game 0 — a green test suite is not a working
submission.
