# Claim to Fame — System Design

Status: pre-round-1 design. Read the three economic results first; they determine the
architecture, and two of them contradict the obvious approach.

---

## 0. The three results that shape everything

### R1 — As issuer, the opponent's `b` is irrelevant in the fair zone

From the payoff table: when `a <= t`, **H gets `a` whether the insurer accepts or
rejects.** Rejection only penalises `I`. So the honest half of our charge revenue does
not depend on any opponent model:

```
R(a) = a * S(a)                        <- fair zone, opponent-independent
     + min(a,c) * (1 - S(a)) * G(a)    <- fraud zone, needs opponent model
```

where `S(a) = P(T >= a | x)` and `G(a) = P(a random opponent accepts a)`.

**Consequence:** ship `a* = argmax a*S(a)` in round 1 with zero opponent data and lose
nothing structural. Opponent modelling is a bonus term, not a prerequisite.

Note also the sign asymmetry: **as issuer our payoff is never negative; as insurer it is
never positive.** Issuer = maximise collection, insurer = minimise leakage. Exploration is
cheap on `a` (downside is forgone revenue) and expensive on `b` (downside is `1.5a` cash).

### R2 — Uncertainty pushes `a` UP and `b` DOWN

For lognormal price beliefs with log-sd `sigma`, `argmax a*S(a)` sits at:

| sigma (log-spread) | optimal `a` percentile |
|---|---|
| 0.3 | ~Q17 |
| 0.4 | ~Q24 |
| 0.6 | ~Q37 |
| 0.8 | ~Q50 |
| 1.0 | ~Q62 |
| 1.2 | ~Q72 |

Estimate `sigma ~= (ln q90 - ln q10) / 2.563`.

Meanwhile `b* = Q33` of the same belief, always (the 2/3 rule), regardless of spread.
So **wide uncertainty moves our charge and our acceptance limit in opposite directions.**
Do not enforce `b >= a` — that invariant is wrong here and will cost money. It is
legitimate to charge 900 and refuse to pay 900 for the same item.

### R3 — `p_valid` kills `b` but does not move `a`

`S(a) = q * S_plus(a)` with `q = p_valid`. Since `q` is a constant multiplier it cancels
out of `argmax a*q*S_plus(a)`: **the optimal charge is independent of `q`.** But
`q < 2/3` forces `b = 0` exactly (no positive price can clear the 2/3 bar).

**Consequence:** on an item we believe is probably not covered, we still charge — the
fraud-zone term is pure upside and there is no penalty for being rejected. Most teams
will submit `a = 0` on items they judge uncovered. That is free money left on the table.
Never submit `a = 0`; the floor is `argmax a*G(a)`.

### Corollary on the cap

`c >= 4t` means the cap essentially never binds for realistic overcharges (charge
`2 * q50` and you are still under `4t` when `t ~= q50`). The cap does not discipline
overcharging — **the opponent's `b` does.** The search grid for `a` can therefore be
`[0, 4 * q50]`, and the cap only matters as an upper clamp.

---

## 1. Layered architecture

Hard separation, three layers. This is the central design decision and the write-up story.

```
files -> [SEMANTIC]  parallel LLM calls, per invoice (not per line)
             |        emits p_valid + unit-price quantiles + evidence
             v
         [PROBABILITY]  q, log-linear quantile interpolation, calibration
             |          emits S(a) = q * S_plus(a), a callable survival function
             v
         [DECISION]   pure numpy, no LLM, no I/O, fully unit-testable
             |        emits (a, b) + the reasoning trace
             v
         submit  ->  observe  ->  update calibrator + opponent models
```

The decision layer must be a pure function of `(S, G, A_dist, c)`. It runs in
microseconds, so it can be re-run offline over every historical round whenever we change
the policy. That replay capability is what makes online learning tractable inside 100
rounds.

---

## 2. The 60-second hot path

Two facts from the brief drive this: **the encrypted zips are published before the round
and only the key is released at T0**, and **later submissions overwrite earlier ones.**

```
T-inf  stage all encrypted zips locally, warm LLM connections, pre-open HTTP sessions
T+0    GET decryption key
T+1    7z decrypt from local disk (no network in the critical path)
T+2    parse invoices.pdf -> line items (id, description, qty, unit)
T+4    fire ALL LLM calls concurrently (asyncio.gather)
T+8    HEURISTIC SUBMIT   <-- safety net, never skipped
T+35   LLM results in (whatever returned), build S(a), optimise
T+45   FULL SUBMIT (overwrites the heuristic)
T+55   hard deadline: whatever is in the buffer goes out
T+60   (post-submit) persist, update models
```

Two rules:

- **Double submit.** The T+8 heuristic submission means a total pipeline failure still
  produces a non-catastrophic answer. Highest value-per-line-of-code in the whole build.
- **Nothing after the POST is on the critical path.** DB writes, calibrator updates, UI
  refresh: all after.

Per-item timeout: any LLM call not back by T+35 is dropped and that item keeps its
heuristic value. Degrade per item, never fail the round.

### Batch per invoice, not per line item

12 line items x 5 agents = 60 calls will not fit in 60s. Instead: 4 calls, each seeing the
**whole invoice** and returning an array of per-item objects. Faster, and it makes
cross-item signals visible at all — duplicate lines, a labour line inconsistent with the
parts lines, invoice total implausible for the described damage.

### Never let the LLM do arithmetic

Ask for **unit-price** quantiles. Multiply by the quantity parsed from the PDF **in code**.
Then cross-check: have one call state the gross total independently, and flag items where
`|llm_total / (qty * llm_unit) - 1| > 0.35`. Unit-vs-total confusion is the most likely
way we submit a number that is wrong by 10x.

---

## 3. Semantic layer — the calls

Four concurrent calls per case, structured output, each seeing the full invoice:

| Call | Input | Output per item |
|---|---|---|
| `validity` | policy + description + invoice + images | `p_valid` (joint covered AND related), `evidence`, `exclusion_hit` |
| `pricing` | invoice + description (+ images) | unit `q10 q25 q50 q75 q90`, `price_basis`, `confidence` |
| `skeptic` | everything | `inflation_flags`, `qty_plausible`, `suggested_multiplier` |
| `parse_check` | invoice only | item ids, qty, unit, restated description |

Ask for the joint `p_valid` directly — do not compute `p_covered * p_related`, they are
strongly dependent. Keep `p_covered` / `p_related` as displayed evidence only.

The skeptic's output is a **multiplicative shrink on the quantile ladder**, clamped to
`[0.5, 1.0]`, not a veto. Vetoes are brittle; shrinkage degrades gracefully.

---

## 4. Probability layer

```python
def survival(p_valid, unit_quantiles, qty, calib) -> Callable[[float], float]:
    """S(a) = P(T >= a). Log-linear interpolation between quantiles,
    lognormal tails beyond q10/q90, scaled by qty, shifted by calib."""
```

- Interpolate in `log(price)` — prices are positive and right-skewed.
- Extrapolate tails as a lognormal fitted to `(q10, q90)`.
- Calibration is **two scalars, not a model**: `mu_shift` (multiplicative bias on the
  median) and `sigma_scale` (are we over- or under-confident). Two numbers are estimable
  from ~20 rounds; an 8-feature `SGDClassifier` is not.

---

## 5. Decision layer

```python
def choose_a(S, G=None, c_cap=None) -> float:   # argmax a*S(a) + min(a,c)(1-S(a))G(a)
def choose_b(S, A_dist=None) -> float:          # Q33 baseline; MC expected cost when A known
```

Grid: 2000 log-spaced points over `[0.05 * q50, 4 * q50]`.

Baseline (no opponent data):

- `a = argmax a * S(a)`
- `b = sup{a : S(a) >= 2/3}`, which is exactly `0` when `p_valid < 2/3`

Strategic (once `G` and the opponent charge distribution are trustworthy):

- `a = argmax [a*S(a) + min(a,c)*(1-S(a))*G(a)]`
- `b = argmin_b E[C(A,b,T)]` by Monte Carlo over `T ~ S` and `A ~ A_dist`

Blend baseline -> strategic with `w = n_obs / (n_obs + 30)`. Never a hard switch.

---

## 6. Learning layer

**Resolve in the first 10 minutes: what does the results endpoint actually expose?** The
whole learning design branches on it.

- **Branch A — per-matchup outcomes exposed.** Each rejection is an interval-censored
  observation on `T`: we paid `1.5a` => `T >= a`; we paid `0` => `T < a`. Accumulate
  `[lo, hi]` bounds per item category and fit `mu_shift` / `sigma_scale` by censored-
  interval maximum likelihood. Beta-Bernoulli buckets on `a / q50` per team give `G`.
- **Branch B — only aggregate net payoff per round.** Per-item labels are impossible.
  Fall back to two global knobs `lambda_a`, `lambda_b` multiplying the baseline outputs,
  hill-climbed on observed round net payoff with a small alternating dither (+-8%, so the
  two stay separable). Crude, converges over 100 rounds, and honest in the write-up.

Assume Branch B until proven otherwise; build so the knobs exist either way.

**The opponent charge distribution is observable regardless of `T`** — incoming `a` values
are data. Track `r = A / q50` per team with mean, sd and a recent EWMA. That feeds
`choose_b`'s Monte Carlo directly and is the cheapest real edge available.

Exploration schedule: rounds 1-10 baseline + collect; 10-30 calibrate; 30-70 exploit with
dither; 70-100 pure exploit, dither off. Dither `a` freely, dither `b` at most +-5%.

---

## 7. Data model (Supabase / Postgres, per `AGENTS.md`)

Append-only. A re-score is a new row, never an update — that is both the audit story and
what makes offline replay possible.

```
rounds        (round_id, case_id, released_at, key_fetched_at, submitted_at, latency_ms, mode)
line_items    (round_id, item_id, description, qty, unit, raw_text)
inferences    (round_id, item_id, model_version, p_valid, p_covered, p_related,
               q10..q90 unit, skeptic_mult, evidence, latency_ms)
decisions     (round_id, item_id, policy_version, a, b, S_at_a, S_at_b, sigma, dither)
outcomes      (round_id, item_id, opponent_team, opponent_a, we_accepted, they_accepted,
               we_paid, we_received)
bounds        (item_id, category, t_lower, t_upper, source_round)
calibration   (as_of_round, mu_shift, sigma_scale, lambda_a, lambda_b, n_obs)
```

The hot path writes an append-only JSONL first; a background task mirrors to Postgres. DB
latency must never touch the submit path. Every new table needs GRANTs **and** RLS
policies — see `AGENTS.md`.

---

## 8. Repo layout (on the existing template)

```
backend/app/
  c2f/
    orchestrator.py     the 60s state machine, double-submit, deadline enforcement
    api_client.py       key fetch + submit, pre-warmed session, retry with jitter
    decrypt.py          7z wrapper
    parsing/            pdf.py  policy.py  images.py
    inference/          validity.py  pricing.py  skeptic.py  schemas.py
    probability/        survival.py  calibration.py
    decision/           optimizer.py  monte_carlo.py
    learning/           bounds.py  opponents.py  knobs.py
    store/              jsonl.py  db.py
  routers/c2f.py        status, history, replay, manual-override endpoints
frontend/src/components/
  RoundMonitor.tsx      live countdown + per-item state
  ItemDecision.tsx      survival curve with a/b reference lines + evidence panel
```

Use the existing `app.llm.get_llm()` factory; do not construct SDK clients in the
pipeline. From `AGENTS.md`: model id is `claude-opus-5` with no date suffix, thinking is
`{"type": "adaptive"}` (`budget_tokens` 400s), and a 200 response can carry
`stop_reason == "refusal"` with empty content.

`decision/` and `probability/` are pure and must have real unit tests — they are the only
place a silent 10x error survives review.

---

## 9. Guardrails before every submit

Non-negotiable, runs in under a millisecond, blocks the POST:

1. every item has both `a` and `b`; no nulls, no NaN
2. `0 <= a <= 4 * q50 * qty` and same for `b`
3. gross total, not unit — assert `a` within `[0.2, 5] * qty * q50_unit`
4. `p_valid < 2/3` => `b == 0` exactly
5. `a > 0` on every item unless `G` says nobody accepts anything (never blanket zero)
6. item ids and item count match the parsed invoice exactly
7. `sum(a)` within `[0.3, 3]` of the independently-stated invoice total
8. **`b >= a` is NOT checked** — see R2
9. any failure => fall back to the last valid payload, log loudly

Default `0` is not a safe fallback for `b`: on a covered item it means paying `1.5a` to
every single opponent. `b` must always be a decision.

---

## 10. Priority cut line

**P0 (before round 1):** stage zips, fetch key, decrypt, parse PDF, one validity call, one
pricing call, log-linear `S(a)`, baseline `a` and `b`, guardrails, double submit. That is a
competitive system on its own.

**P1:** skeptic call, persistence, opponent charge stats, the two calibration scalars, the
decision UI.

**P2:** Beta-Bernoulli acceptance model, Monte Carlo `b`, opponent-aware `a`.

**P3 (probably never):** contextual bandits, Thompson sampling, per-category calibration.

Do not build: custom nets, RL, PyMC, fine-tuning, a generic fraud classifier with no
labels, a vector DB.

---

## 11. Judging

Assessed on approach plus performance, with a write-up, and the best teams present. Per
`TRACK.md`, QuantCo values decisions a human can *check*.

The demo: take one line item, show the survival curve with `a` and `b` as reference lines,
then drop `p_valid` below 2/3 live and watch `b` snap to zero while `a` does not move. That
is R3 made visible, the audience can predict it before it happens, and it survives a
hostile question because it is a theorem rather than a heuristic.

The write-up's spine is the layer separation: semantic reasoning != probability estimation
!= decision making. Everything else is detail.

---

## 12. What case 0 taught us

Case 0 is a single line item, `New Bike`, quantity 1. The whole case turns on
policy section 4: the insurer reimburses **the market value at the time of the
theft**, and the description states that value as EUR 420. Cover conditions are
met (locked to a lamp post) and no exclusion applies. So `t = 420`, and an
invoice line reading "New Bike" is worth EUR 420 rather than a new bicycle's
price. QuantCo's own starter script hints at it: its placeholder values are
`charge_price = 410`, `acceptance_limit = 430`.

Scored offline against a field of five plausible opponents (`decision/payoff.py`,
which reproduces the brief's worked example exactly):

| strategy | a | b | net |
|---|---|---|---|
| oracle (knows `t`) | 420.00 | 420.00 | **+1280** |
| ours, confident belief | 418.65 | 420.53 | **+1273** |
| the `c2f/` engine's mock belief (380/420/450) | 402.50 | 410.55 | +983 |
| ours, tight belief (+-6%) | 382.83 | 413.99 | +884 |
| ours, heuristic fallback | 356.46 | 45.35 | +552 |
| ours, loose belief (+-25%) | 330.02 | 389.66 | +420 |
| ours, missed the clause | 659.33 | 692.63 | -161 |
| naive, prices a new bike | 800.00 | 900.00 | -820 |

Three conclusions, all of which changed the code:

**The decision layer is not the bottleneck.** Given a correct belief it lands
within 0.5% of an oracle that knows `t`. Every remaining euro is in the semantic
layer. Effort goes there.

**The pricing call needs the policy.** An earlier version withheld it, reasoning
that a policy says nothing about market rates. That reasoning is wrong: the
basis-of-indemnity clause *is* the price, and it outweighs any market estimate.
Withholding it is the difference between +1273 and -161 on this case. The pricing
prompt now leads with "find the basis of indemnity and apply THAT basis, not the
wording of the invoice line".

**A stated value must collapse the spread.** The `tight (+-6%)` row loses EUR 396
against the oracle, and EUR 210 of that is one error: `b = 413.99` sits EUR 6
below `t`, so it wrongfully rejects a fair EUR 420 charge and pays the 1.5x
penalty. When the documents state the amount, hedging the range is not caution -
it is the expensive choice. The prompt now asks for a near-zero spread in that
case, and the `confident` row is what that produces. This applies to any
implementation using a 1/3-quantile rule, including the `c2f/` engine: its
triangular belief puts `b` below its own mid by construction, so a spread wider
than reality is what costs it the 210.

## 13. Confirmed API constraints

- Line items are addressed by `index`, the invoice POS column. Never renumber.
- **There is no results endpoint.** List games, fetch key, submit - that is all.
  Per-matchup outcomes are not exposed, so Branch A of section 6 is unavailable
  and the two global knobs are the learning mechanism. The leaderboard is the
  only feedback channel.
- Omitted line items default to `0 / 0` and still participate, so every parsed
  item must appear in the payload. `b = 0` on a covered item pays the penalty to
  every opponent.
- 101 games exist (0-100). Game 0 is a permanent test game.
- Archives are AES-256 zips. We read them with `pyzipper` rather than shelling
  out to 7z: no per-machine install, no subprocess in the 60-second window. 7z is
  not installed on at least one team machine, so the subprocess path fails there.

## 14. Two engines currently live in this repo

| | `backend/app/c2f/` (this design) | `c2f/` (the other engine) |
|---|---|---|
| LLM calls | 3 concurrent, structured | 1 call, fast+full passes |
| belief | 5-knot ladder, log-space, lognormal tails | triangular `t_low/t_mid/t_high` |
| `a` | `argmax a*S(a)` | `t_mid * (1 - 0.25 * spread)`, floored at `t_low` |
| `b` | `Q(1 - (2/3)/p_valid)`, exact | `tri_quantile(.., 1/3)` |
| decrypt | `pyzipper`, in-process | subprocess |
| entry point | `python -m app.c2f.run` | `make <game_id>` |
| offline scorer | `decision/payoff.py` | none |

They agree on the economics that matter - the 2/3 rule, `b = 0` when uncovered,
still charging on an uncovered item - which is a good sign for both.

Two differences are substantive rather than cosmetic:

* `a = t_mid * (1 - K * spread)` can never exceed `t_mid`. That forgoes R2: when
  our belief is genuinely wide the revenue-maximising charge sits *above* the
  median, and it forgoes the fraud-zone term entirely.
* Only this design has an offline scorer, so only this one can answer "would that
  change have made us money on the cases we have already seen".

**Pick one before a live round.** Two submit paths for one team, with
last-write-wins semantics and a 60-second window, is a way to lose a round to a
race between our own processes.
