# Claim to Fame engine — architecture

The challenge brief is in `GAME_DESCRIPTION.md`, the API in `API_HANDBOOK.md`.
Code lives in `c2f/`, tests in `tests/`, per-game logs in `runs/`.

## One command per game

```
make 7          # play game 7: wait for key → decrypt → extract → model → price → submit
make check      # real model against the permanent test game 0
make mock       # canned model answer, tests the plumbing
make test       # unit tests
```

Start `make N` a few seconds before the game opens. `get_case.sh` polls the key endpoint
every 0.3 s; the 60-second clock starts when the key appears. Everything must be submitted
within that minute. Last write wins on the server.

## Pipeline

```
IN              EXTRACT            MODEL                 PRICE              OUT
cases/NN.zip    extract.py         llm.py                price.py           submit.py
+ key   ──►     case dict   ──►    one LLM call   ──►    (a, b) per item ──► PUT /submissions
get_case.sh     policy text        covered? related?     a = charge         runs/game_NN.json
                description text   t_low/mid/high        b = accept limit
                invoice rows       clause + reason
                images (b64)
```

| Step | Module | Input | Output |
|---|---|---|---|
| IN | `get_case.sh` | `cases/case_NN.zip`, `GET /api/games/NN/key` | `cases/case_NN/` |
| EXTRACT | `c2f/extract.py` | policy.txt, description.txt, invoices.pdf, *.png | case dict (below) |
| MODEL | `c2f/llm.py` | case dict + system prompt | JSON: per item covered, related, clause, t_low/t_mid/t_high, t_if_covered, reason; policy_summary |
| PRICE | `c2f/price.py` | model JSON | `[{index, charge_price, acceptance_limit}]` |
| OUT | `c2f/submit.py` | rows | `PUT /api/games/NN/submissions`, log in `runs/` |

`c2f/run.py` orchestrates. It runs two model passes in parallel: a **fast** one (small
model) submitted as soon as it lands, and a **full** one that overwrites it if it arrives
before ~53 s after decrypt. If both fail, nothing is submitted and the exit code is 1 —
fall back to `./submit.sh NN 1:A:B ...` by hand.

## How the input is structured

The model sees one object, built by `extract.py`:

```json
{
  "game_id": 0,
  "policy":      "BICYCLE THEFT INSURANCE ... (full text, capped at 30k chars)",
  "description": "Client came back from work ... the bike was worth 420 Euros.",
  "invoice_text": "full text of invoices.pdf",
  "invoice_meta": {"trade": "Bikeshop", "vendor": "Bikey Bike Ltd", "date": "6 Jan 2026"},
  "items": [{"index": 1, "description": "New Bike", "quantity": 1, "unit": "unit"}],
  "images": [{"name": "...", "media_type": "image/png", "b64": "..."}]
}
```

**Policy and description are passed verbatim; the model does the reading.** They are
short (a policy is ~1.5 KB), and the work — "is a new bike covered under clause 4 given
clause 3 and the description says it was locked to a lamp post?" — is cross-referencing
natural language, which is what the model is for. The description also carries price
signal ("worth 420 Euros") that the model must see next to the invoice.

The invoice is parsed deterministically (`parse_items`: `POS | DESCRIPTION | AMOUNT | UNIT`,
wrapped descriptions handled) *and* sent as raw text. The model's item indices are
validated against the parse when it succeeded; missing indices are logged and priced as
unknown.

Prompt layout (`llm.build_user_message`):

```
<policy> ... </policy>
<damage_description> ... </damage_description>
<invoice trade="Bikeshop" vendor="Bikey Bike Ltd"> raw pdf text </invoice>
<parsed_line_items> 1 | New Bike | 1 unit </parsed_line_items>
[images attached]
```

The system prompt (`llm.SYSTEM`) asks, per item: covered? related? fair **gross total**
(qty × unit price, incl. 19 % VAT) as low / mid / high; 0 if not covered, with the
would-be price in `t_if_covered`; cite the clause; return JSON only.

## Pricing rule (`price.py`)

Secret fair value `t` per item. Payoffs: charging `a ≤ t` is always paid; `a > t` is paid
only by opponents whose limit `b' ≥ a`. Accepting a fair charge costs 1×, wrongly
rejecting costs 1.5×, accepting fraud costs 1× (capped).

- **b (acceptance limit)** = 1/3-quantile of a triangular(t_low, t_mid, t_high) belief.
  Accept iff `P(t ≥ a') > 2/3`, from the 1.5× vs 1× asymmetry.
- **a (charge)** = `t_mid × (1 − 0.25 × spread)`, floored at `t_low`,
  `spread = (t_high − t_low) / t_mid`. Sits just under the estimate; lower when unsure.
- **Not covered / not related**: `b = 0`; `a = 0.6 × t_if_covered` (no downside under the
  rules: rejected fraud costs the issuer nothing).

Constants at the top of `price.py`: `K_UNCERTAINTY`, `UNCOVERED_CHARGE`, `B_QUANTILE`.

## Configuration (`.env`)

| Var | Meaning |
|---|---|
| `TEAM_API_KEY` | QuantCo team key (required) |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | whichever is set picks the provider (Anthropic first) |
| `C2F_MODEL` | full-pass model (default `claude-opus-5` / `gpt-5`) |
| `C2F_FAST_MODEL` | fast-pass model (default `claude-sonnet-5` / `gpt-5-mini`) |
| `C2F_REASONING` | OpenAI `reasoning_effort` for gpt-5 models (default `low`) |

## Logs

`runs/game_NN.json`: the case dict (minus image bytes), both model outputs with timings,
the priced rows, every submission with the server response. This is the material for the
strategy write-up.

## Not built (deliberately)

Scheduler, leaderboard feedback loop, dashboard, database. Manual `make N` per game.
