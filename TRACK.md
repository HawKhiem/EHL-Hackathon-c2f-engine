# Track: QuantCo

> **Everything below was written before the challenge brief was released.** The design
> language is safe — it comes from the partner's public identity. The positioning is an
> *inference*: check it against `CHALLENGE.md` and the on-site mentors, and when they
> conflict, the brief wins. Do not defend a guess made two days early.

## This branch

| | |
|---|---|
| Supabase project | `ehl-quantco` (own database, isolated from the other branches) |
| Supabase API / Studio | `54421` / `54423` |
| Backend | `8000` |
| Dev server | `5173` |

Isolated ports mean you can run this branch and another side by side without a clash,
and migrations here can never leak into another track's database.

## Positioning (inference — validate)

Data-first and trustworthy. The differentiator is not that a model produced an answer;
it is that a human can **check** the answer. Lead with evidence, provenance and decision
logic. Treat visual flourish as a liability.

Angles worth putting to a mentor in the first ten minutes:

- **Verifiable decisions** — every output carries its inputs, its weights and the
  thresholds applied, so a reviewer can reconstruct it without trusting the system.
- **Where the model is unsure** — confidence framed as distance from a decision boundary
  is far more actionable than a bare probability, and it routes work to humans well.
- **Auditability as a feature** — versioned model ids, reproducible scores, and a summary
  a reviewer can paste into a case file.

## Design language

| | |
|---|---|
| Feel | Minimal, professional, dense but legible |
| Palette | Desaturated slate-blue on near-neutral; colour reserved for signal |
| Type | Strong hierarchy, `tabular-nums` on every figure so columns align |
| Radius | Tight (`0.375rem`) — reads as instrument, not consumer app |
| Layout | Metric cards, real tables, an explicit "why" panel beside every verdict |
| Charts | Recharts. Attribution bars, reference lines at thresholds. No 3D, no gradients |
| Avoid | Hero gradients, big illustrations, emoji, anything that reads as marketing |

Rule of thumb: if a number and a chart compete for the same space, the number wins.

## At kick-off

1. Paste the real brief into `CHALLENGE.md`, verbatim, including judging criteria.
2. Ask a mentor: **which decision does the end user distrust today, and why?** That
   answer is your product.
3. Design the schema around the decision you must justify. Persist decisions
   append-only — a re-score is a new row, never an update. "Here is every call the
   system made and why" is the demo.
4. Build one decision path end to end before adding a second.

Remember: every new table needs `grant` statements *and* RLS policies. See `AGENTS.md`.

## Optional tooling

```
/plugin marketplace add nextlevelbuilder/ui-ux-pro-max-skill
/plugin install ui-ux-pro-max@ui-ux-pro-max-skill
```

Lower priority on this track than on the other two — restraint matters more than polish
here, and the skill pushes toward richer visuals. Useful for the chart and table work.

## Demo trap

A dashboard of plausible numbers is easy to build and easy to dismiss. What lands is
changing one input live and having the attribution move the way the audience predicted.
Rehearse that exact moment, and make sure it survives a hostile question.
