# Claim to Fame engine — architecture

The challenge brief is in `GAME_DESCRIPTION.md`, the API in `API_HANDBOOK.md`.
Code lives in `c2f/`, tests in `tests/`, per-game logs in `runs/`.

## One command per game

```
make 7          # play game 7: wait for key → decrypt → extract → model → price → submit
make check      # real model against the permanent test game 0
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
                image names
```

| Step | Module | Input | Output |
|---|---|---|---|
| IN | `get_case.sh` | `cases/case_NN.zip`, `GET /api/games/NN/key` | `cases/case_NN/` |
| EXTRACT | `c2f/extract.py` | policy.txt, description.txt, invoices.pdf, *.png | case dict (below) |
| DIGEST | `c2f/policy.py` | policy.txt | `policy_digest`: limits / deductibles / exclusions |
| MODEL | `c2f/llm.py` | case dict + system prompt | JSON: per item covered, related, clause, t_low/t_mid/t_high, t_if_covered, reason; policy_summary |
| PRICE | `c2f/price.py` | model JSON | `[{index, charge_price, acceptance_limit}]` |
| OUT | `c2f/submit.py` | rows | `PUT /api/games/NN/submissions`, log in `runs/` |

`c2f/run.py` orchestrates. **One pass**, on one model (`gpt-5.6-terra` everywhere). Its
timeout is whatever is left of the ~53 s clock (floor `MIN_MODEL_S`); the invoice is split into
`ceil(n / C2F_CHUNK_ITEMS)` parallel calls (10 items → 1, 20 → 2, 32 → 4, capped at
`C2F_MAX_CHUNKS`) so a slow chunk costs only its own rows, and every submission still carries
the full row set. Because one model and one pass free up the call budget, the pass runs at
`reasoning_effort` **medium**: 29 items in 12 s, a 22-item case in 29 s, both inside the clock.
**Every chunk submits.** Last write wins and only the state at close is scored, so an
intermediate row still at its 0/0 placeholder costs nothing once a later chunk overwrites it,
while a board that is never empty survives a crash, a hang or a misjudged clock. 0/0 is also
the right blind guess: `b=0` wrongly rejects fair charges at `0.5a` extra each, where `b=∞`
would accept fraud up to the cap `c ≥ 4t`. If it fails, nothing is submitted and the exit
code is 1 — fall back to `./submit.sh NN 1:A:B ...` by hand.

The old **fast** safety pass (a second, smaller model submitted the moment it landed, never
aggregated) is off by default and available behind `--fast`. It was insurance from when the
full pass ran on `gpt-5.6-sol` at 7–35 s; on terra the single pass lands in 2–7 s, so the
second call bought a duplicate answer rather than safety.

## How the input is structured

The model sees one object, built by `extract.py`:

```json
{
  "game_id": 0,
  "policy":      "BICYCLE THEFT INSURANCE ... (full text, never truncated)",
  "description": "Client came back from work ... the bike was worth 420 Euros.",
  "invoice_text": "full text of invoices.pdf",
  "invoice_meta": {"trade": "Bikeshop", "vendor": "Bikey Bike Ltd", "date": "6 Jan 2026"},
  "items": [{"index": 1, "description": "New Bike", "quantity": 1, "unit": "unit"}],
  "images": [{"name": "...", "media_type": "image/png"}]
}
```

**Case photos are skipped.** `*.png/*.jpg` in the case folder are listed by name in the case
dict but never read or attached to the model call: the coverage and pricing decisions come from
the policy, the description and the invoice, and image upload costs time inside the 60 s window.

**Policy and description are passed verbatim; the model does the reading.** The work —
"is a new bike covered under clause 4 given clause 3 and the description says it was
locked to a lamp post?" — is cross-referencing natural language, which is what the model
is for. The description also carries price signal ("worth 420 Euros") that the model must
see next to the invoice.

**Nothing is truncated.** Policies run 40–65 KB — under 20k tokens, a rounding error
against the context window — and cutting one drops the tail, which is where the exclusions
and caps live.

They *are* long enough to bury the decisive clauses, so `c2f/policy.py` runs a
pre-extraction pass on every game: one call to the same model (`gpt-5.6-terra`, override `C2F_DIGEST_MODEL`) reads policy.txt and returns the parts that
bind — insured event, conditions, limits, deductibles, exclusions, obligations — rendered
into a `<policy_digest>` block placed *in front of* the verbatim text. It is a reading aid,
never a replacement: the full policy is still in the prompt and the prompt says so.

Best-effort and bounded. `run.py` starts it in parallel with the **fast** pass (which must
never wait — it is the safety submission) and gives the **full** passes `DIGEST_WAIT_S = 10`
to pick it up; on timeout or API failure they go with the verbatim policy alone, which is
complete. `policy.build()` returns rather than mutates so the main thread owns the assign
and cannot race a prompt already being built. `backtest.py` attaches it too, so a replay
builds the same prompt as a live run.

The invoice is parsed deterministically (`parse_items`: `POS | DESCRIPTION | AMOUNT | UNIT`,
wrapped descriptions handled) *and* sent as raw text. The model's item indices are
validated against the parse when it succeeded; missing indices are logged and priced as
unknown.

Prompt layout (`llm.build_user_message`):

```
<market_history> ... </market_history>   (c2f/history.py: t brackets the market actually accepted
                                        in past rounds, by category and by repeated label; omitted
                                        if there are no truth files)
<policy_digest> ... </policy_digest>   (limits/exclusions pulled out; omitted if that call failed)
<policy> ... </policy>
<damage_description> ... </damage_description>
<invoice trade="Bikeshop" vendor="Bikey Bike Ltd"> raw pdf text </invoice>
<parsed_line_items> 1 | New Bike | 1 unit </parsed_line_items>
```

The system prompt (`llm.SYSTEM`) asks, per item: covered? related? fair **gross total**
(qty × unit price, incl. 19 % VAT) as low / mid / high; 0 if not covered, with the
would-be price in `t_if_covered`; cite the clause; return JSON only.

## Pricing rule (`price.py`)

Secret fair value `t` per item. Payoffs: charging `a ≤ t` is always paid; `a > t` is paid
only by opponents whose limit `b' ≥ a`. Accepting a fair charge costs 1×, wrongly
rejecting costs 1.5×, accepting fraud costs 1× (capped).

Belief on `t`: lognormal with median `t_mid × bias` and log-sd `max(σ, model spread)`.
`bias`, `σ` (and the market's acceptance of over-charges `p0`, `k`) are learned by
`c2f.calibrate` from the `[t_lo, t_hi)` bounds `c2f.truth` recovers after every game
(`runs/calibration.json`, read at pricing time; defaults `bias 1, σ 0.4, p0 0.35, k 2`).

- **b (acceptance limit)** = 1/3-quantile of the belief.
  Accept iff `P(t ≥ a') > 2/3`, from the 1.5× vs 1× asymmetry.
- **a (charge)** maximises `mean − 0.75·sd` of the per-opponent payout: `a` if `a ≤ t`, else
  `a × p0 × (a/t)^−k` (the fraction of reviewers still accepting an over-charge). A fair
  charge is paid by all 16 opponents, a fraudulent one by ~a third → the optimum sits
  ~0.5–0.7× the median, lower the less sure we are (the sd term; pure expectation would
  chase the upper tail).
- **Not covered / not related**: `b = 0`; `a = 0.6 × t_if_covered` (no downside under the
  rules: rejected fraud costs the issuer nothing).

**No votes.** An earlier version ran the fast pass plus three full-model votes and aggregated
them into an ensemble. Comparing fast against full over six logged games
(`runs/game_{01,03,04,06,07,08}.json`, scored on the `runs/truth_game_*.json` brackets):
the two agreed on 70 of 82 coverage calls, and of the 12 disagreements 4 favoured fast,
4 favoured full and 4 were undecidable — a coin flip. On price the full pass *was* clearly
better (median |log error| 0.23 vs 0.41 over the 46 items both called covered), so the full
pass alone decides the numbers and the fast one is kept only as the fallback submission.

Constants at the top of `price.py`: `RISK_AVERSION`, `UNCOVERED_CHARGE`, `B_QUANTILE`.

## Configuration (`.env`)

| Var | Meaning |
|---|---|
| `TEAM_API_KEY` | QuantCo team key (required) |
| `OPENAI_API_KEY` | required — OpenAI is the only provider `c2f/llm.py` and `c2f/policy.py` support |
| `C2F_MODEL` | full-pass model (default `gpt-5.6-terra`) |
| `C2F_FAST_MODEL` | fast-pass model (default `gpt-5.6-terra`) |
| `C2F_DIGEST_MODEL` | policy-digest model (default `gpt-5.6-terra`) |
| `C2F_REASONING` | OpenAI `reasoning_effort` for the pass (default `medium`; gpt-5.6 floor is `none`, not `minimal`) |
| `C2F_REASONING_FAST` | effort for the optional `--fast` pass (default `none`) |
| `C2F_CHUNK_ITEMS` | items per parallel call (default 10) |
| `C2F_MAX_CHUNKS` | cap on parallel calls (default 8) |

Every one of these is written out in `.env` with the code default made explicit; delete a line
and the default in `c2f/` applies.

## Logs

`runs/game_NN.json`: the case dict, both model outputs with timings,
the priced rows, every submission with the server response. This is the material for the
strategy write-up.

## Backtest (gate for algorithm changes)

`c2f/backtest.py` replays the current strategy on past decrypted cases and scores it against
the real opponents of that round using the public leaderboard (their charges, their implied
accept limits, the inferred `t` bounds). Output: actual vs replayed net and rank per game,
pessimistic and optimistic where outcomes are open, `runs/backtest/summary.json`.
The verdict — a positive expected net in a majority of games plus a positive total; money is the
only gate, rank is reported (top-3 = `RANK_TARGET`) but never fails a run — is always over the
**last 10** old games (`WINDOW`) = the 10 most recent completed
games with a decrypted case; `make backtest G=...` replays only those games and re-scores the rest
from their stored replays, and an old game with no stored replay counts as not won and is listed
as `missing`. Running it is a judgement call, not a gate — a pre-commit hook used to enforce it
and was removed as too much friction. See `c2f/README.md`.

## Not built (deliberately)

Scheduler, leaderboard feedback loop, dashboard, database. Manual `make N` per game.
