# Claim to Fame — our approach (team AsianSuperNerds)

A short write-up of how we played, what the data taught us, and why it went the way it did.
No claim data (policies, descriptions, invoices, run logs that embed them) is checked into this
repository — it lives only on the playing machine under `cases/` and `runs/`, both gitignored.

## The problem as we ended up seeing it

Each round, every team sets a charge `a` and an acceptance limit `b` per invoice line. A charge
at or under the secret fair value `t` is paid by **every** reviewer (refusing a fair charge costs
the refuser 1.5×); a charge above `t` pays only from reviewers whose `b` is still above it, and
measured over the field that revenue is flat at ~0.5 t regardless of how far over you go.
Accepting costs `a`; refusing a fair charge costs `1.5a`. So:

- the issuer's payoff is a **step** — `a` if `a ≤ t`, ~0.5 t above — and the right charge is the
  highest value you still believe is fair;
- the reviewer's rule is **accept iff P(fair) > 2/3**, i.e. `b = Q(1/3)` of an *honest* belief;
- the money is in the one or two biggest items of a round (whales) and in simply being present:
  a missed round costs more than any pricing mistake.

## Architecture (in `c2f/`)

1. **Extract** — decrypt the case, pypdf the invoice, recover POS numbers + line labels with a
   layout-only parser (no unit vocabulary; the one that had one lost a whole round to "68 lines").
2. **Estimate** — one LLM call (OpenAI) per case, chunked in parallel above 10 items so the first
   board is on the server in ~15–20 s; a board is submitted after every chunk (last write wins).
3. **Price** — turn the estimate into a lognormal belief on `ln t`, then `a` = argmax of the
   measured step payoff, `b` = the 2/3 quantile rule with coverage probability inside it,
   whale-specific floors on `b`.
4. **Learn** — after every round, infer `[t_lo, t_hi)` per item from the public payouts and
   matchup cells (`truth`), refit the calibration (`calibrate`, with a live reliability monitor of
   `P(t ≥ b)`), and feed the proven brackets back as **market memory**.
5. **Autoplay** — read the schedule, launch each round 90 s before it opens, play with the v2
   engine, never re-play a logged round. Presence is the whole job.

## The two engines

**v1** — the LLM prices everything (`t_low/t_mid/t_high`), a 3-segment log-log spline corrects
its size-dependent bias, the belief's σ comes from interval-censored MLE. Its objective evolved
from mean − λ·sd (too timid: ~Q(0.27), 21–40 % of provably collectable income forfeited) to pure
expected value on the step payoff.

**v2** — built from a data pass over rounds 30–45, which found that for line items that recur
across rounds the market's own history predicts `t` with MAD ≈ 0.08–0.10 in log space, versus
0.36–0.41 for the LLM on the same items, and that memory covers ~2/3 of items. So: a per-line
**memory prior** (precision-weighted against the LLM — equal weighting is worse than memory
alone), a 1,400-character prompt with per-item memory anchors inline, `p_covered` instead of a
boolean, honest quantiles, three parallel samples with a per-item median, first sample submitted
as insurance. One call, 5–16 s. On rounds 48–53 it placed 1st three times.

## What the data said that we did not expect

- `t` is exogenous (it sits at the 63rd percentile of the field's charges, IQR 0.33–0.86).
- The LLM's stated interval is uninformative about its own error (corr −0.03); its level drifts
  with every prompt edit; memory does not.
- The best team's lead was two whale items charged at ~2× the field median and still fair.
- "Accept everything" breaks even; the winners accept *the right* charges (the team with a 93 %
  accept rate paid 25 k in fraud in one round).
- Widening σ to match measured residuals **loses** money (−80 k on a leave-one-out refit): the
  decision-relevant σ is not the residual σ, because the payoff is asymmetric and bounded.
- Memory must carry **price, not coverage**: a past refusal under another policy predicts the
  next case's coverage only 40 % of the time, and one refused air-con unit read as a "€47–77"
  price anchor cost a round −48.6 k.

## Why we succeeded / did not

- **Succeeded:** the measured-payoff maths (step objective, derived `b`, coverage inside the
  rule) and the market memory both paid, immediately and measurably — v2 took three 1st places
  in its first six rounds, at a fifth of v1's latency.
- **Succeeded:** the learning loop was real — truth inference from public payouts, a calibration
  refit and a reliability alarm after every round — so every bad round produced a number, not
  an opinion, and most fixes were validated on a two-sided counterfactual before going live.
- **Did not:** operations cost more than modelling. Two rounds were lost to nobody pressing the
  button, one to a Windows path/7z chain, one to a 33 s model call against a 60 s window —
  ~120 k, more than all pricing errors of the same window combined. Autoplay came late.
- **Did not:** the LLM's level error on first-sight items (≈ +0.36, MAD 0.8 live) is the residual
  we never fixed; no constant can, only more memory coverage or a prior for novel items.
- **Did not:** we tuned constants on one window several times (replay vs live, old vs new
  prompt) and each time the window moved under us. The lesson was to measure on live rounds
  only, with a drift alarm — which we built, but in the last hours.
- **Team process:** several engines were developed in the same working tree by people on
  different machines; generated artefacts collided on every pull and two machines once played the
  same round. A one-line rule ("one machine plays, artefacts are not committed") would have saved
  a day.
