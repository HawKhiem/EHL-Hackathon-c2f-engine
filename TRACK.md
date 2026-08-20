# Track: Viktor

> **Everything below was written before the challenge brief was released.** The design
> language is safe — it comes from the partner's public identity. The positioning is an
> *inference*: check it against `CHALLENGE.md` and the on-site mentors, and when they
> conflict, the brief wins. Do not defend a guess made two days early.

## This branch

| | |
|---|---|
| Supabase project | `ehl-viktor` (own database, isolated from the other branches) |
| Supabase API / Studio | `54521` / `54523` |
| Backend | `8001` |
| Dev server | `5174` |

Isolated ports mean you can run this branch and another side by side without a clash,
and migrations here can never leak into another track's database.

## Positioning (inference — validate)

Make complicated automation feel simple. The product is not "an AI that does work" — it
is **trustable delegation**: you can see what it is doing, stop it, and approve the
consequential steps.

Angles worth putting to a mentor in the first ten minutes:

- **The approval gate is the product, not a limitation.** Automation people actually
  adopt is automation they can interrupt. Make the human-in-the-loop step feel like
  control, not friction.
- **Visible business outcome.** "Drafted 14 replies, 12 approved, 2 edited" beats any
  description of the model.
- **Live progress.** Watching work happen is what makes it feel like an employee rather
  than a form submission.

## Design language

| | |
|---|---|
| Feel | Friendly, modern, polished, approachable |
| Palette | Warm indigo on soft surfaces; status colour is load-bearing |
| Radius | Generous (`0.75rem`) — calm, consumer-grade |
| Layout | Timeline / activity feed, approve-reject gates, status chips, result summaries |
| Motion | Use it for state transitions only. A row changing status should be noticeable |
| Avoid | Dense tables, jargon, raw JSON on screen, anything that looks like a config panel |

Status states must be distinguishable at a glance: waiting, running, done, failed.

## At kick-off

1. Paste the real brief into `CHALLENGE.md`, verbatim, including judging criteria.
2. Ask a mentor: **which task would a real user delegate but not blindly trust?** That
   gap is where the approval gate earns its place.
3. Model state as explicit rows and let Supabase **Realtime** push changes to the feed —
   do not poll. Every state change being a row write is what makes the UI feel live.
4. Build one task type end to end, including the reject path. A demo that only shows the
   happy path invites exactly the question you cannot answer.

Remember: every new table needs `grant` statements *and* RLS policies, and Realtime needs
the table added to the publication. See `AGENTS.md`.

## Optional tooling

```
/plugin marketplace add nextlevelbuilder/ui-ux-pro-max-skill
/plugin install ui-ux-pro-max@ui-ux-pro-max-skill
```

**Recommended on this track** — UI polish is a genuine differentiator when the pitch is
"automation that feels simple".

## Demo trap

An activity feed full of green ticks proves nothing; it looks staged. Show the agent
pausing for approval, a human rejecting something, and the system handling it gracefully.
The reject path is what makes the happy path believable.
