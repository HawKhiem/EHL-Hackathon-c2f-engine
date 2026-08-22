# Claim to Fame engine — design (2026-08-22)

One command per game: `pixi run python -m c2f.run GAME_ID`. Must finish in < 60 s.

## Pipeline

IN → EXTRACT → MODEL → PRICE → OUT

| Step | Module | Input | Output |
|---|---|---|---|
| IN | `get_case.sh` (existing) | `cases/case_NN.zip` + key from `GET /api/games/NN/key` | `cases/case_NN/` |
| EXTRACT | `c2f/extract.py` | policy.txt, description.txt, invoices.pdf, *.png | `case` dict: policy, description, invoice_text, items[{index, description, quantity, unit}], images[] |
| MODEL | `c2f/llm.py` | case dict + rules prompt | per item: covered, related, clause, t_low/t_mid/t_high, reason; plus policy_summary |
| PRICE | `c2f/price.py` | model output | per item: a (charge), b (acceptance limit), gross totals |
| OUT | `c2f/submit.py` | [{index, a, b}] | `PUT /api/games/NN/submissions`; copy in `runs/game_NN.json` |

## Why AI-native

Policy and description go into the prompt verbatim. The judgement (covered? related?
fair price?) is cross-referencing natural language, which is what the model is for.
The deterministic invoice parse is a helper; the raw invoice text is also sent, and the
model's item indices are validated against the parse when it succeeded.

## Pricing rule

- `b` = 1/3-quantile of the t range (accept iff P(t ≥ a') > 2/3, since wrongful
  rejection costs 1.5× and accepting fraud costs 1×). Triangular(t_low, t_mid, t_high).
- `a` = t_mid × (1 − k·spread), spread = (t_high − t_low)/t_mid, k = 0.25, floored at t_low.
- Not covered / not related: b = 0; a = small plausible price (UNCOVERED_CHARGE × t_mid
  guess, or 0 if the model gives no guess). No downside under the rules.

## Time budget and fallback

- Decrypt ≈ 3 s, extract < 1 s, model ≤ 35 s (hard timeout), price+submit < 1 s.
- If the model fails/times out: submit heuristic a = b = 0 only for items we cannot price;
  covered-looking items never get b = 0 if any estimate exists.
- Provider auto-selected: ANTHROPIC_API_KEY → Claude; else OPENAI_API_KEY → OpenAI;
  `--mock` uses a canned answer for testing.

## Out of scope (for now)

Scheduler, leaderboard feedback loop, dashboard, database.
