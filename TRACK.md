# Track: Cognition

> **Everything below was written before the challenge brief was released.** The design
> language is safe — it comes from the partner's public identity. The positioning is an
> *inference*: check it against `CHALLENGE.md` and the on-site mentors, and when they
> conflict, the brief wins. Do not defend a guess made two days early.

## This branch

| | |
|---|---|
| Supabase project | `ehl-cognition` (own database, isolated from the other branches) |
| Supabase API / Studio | `54621` / `54623` |
| Backend | `8002` |
| Dev server | `5175` |

Isolated ports mean you can run this branch and another side by side without a clash,
and migrations here can never leak into another track's database.

## Positioning (inference — validate)

Developer-facing, and the audience is the hardest to impress: they have used coding
agents and know exactly where they fail. The credible gap is not capability, it is
**trust** — knowing what the agent did, why, and being able to intervene before it goes
wrong.

Angles worth putting to a mentor in the first ten minutes:

- **Observability over autonomy.** An agent you can watch and correct beats a more
  capable one you cannot.
- **Intervention, not just cancellation.** Pausing mid-run, correcting course, and
  resuming is meaningfully different from killing the job and starting over.
- **Show the reasoning trail.** Which files it read, what it decided, what it changed —
  a reviewable trace is the artefact.

## Design language

| | |
|---|---|
| Feel | Technical but clean; efficient, never crowded |
| Palette | Graphite surfaces, one precise accent. Dark is the primary experience |
| Radius | Sharp (`0.25rem`) |
| Type | Mono for identifiers, paths, diffs and traces. Proportional for prose |
| Layout | Step timeline, streaming output, code/diff views, relationship graphs |
| Colour rule | In a trace, colour means "state changed" — never decoration |
| Avoid | Marketing gloss, oversized headings, gratuitous animation, fake terminal chrome |

## At kick-off

1. Paste the real brief into `CHALLENGE.md`, verbatim, including judging criteria.
2. Ask a mentor: **what does an engineer do today when the agent goes wrong?** That
   recovery loop is the product.
3. Stream the run rather than returning it. The SSE pattern in `app/routers/llm.py` and
   `lib/api.ts` is the reference — reuse it for agent steps, and keep step state in the
   database so a reload does not lose the trace.
4. Build pause-and-resume before adding a second capability. It is the hard part and the
   whole point.

Remember: every new table needs `grant` statements *and* RLS policies. See `AGENTS.md`.

## Optional tooling

```
/plugin marketplace add nextlevelbuilder/ui-ux-pro-max-skill
/plugin install ui-ux-pro-max@ui-ux-pro-max-skill
```

```bash
npm i -g @colbymchenry/codegraph
codegraph install && codegraph init
```

**Both recommended on this track.** `codegraph` is doubly relevant here: it makes your own
agent faster at navigating the codebase, and a code-graph view is on-theme for the demo.
It runs a local background daemon and a bundled Rust kernel — heavier than a normal npm
CLI, so install it early rather than mid-build. `.codegraph/` is already gitignored.

## Demo trap

This audience will not be impressed by an agent writing code — they have seen that. They
will be impressed by an agent they can stop, question and redirect. Show it going wrong
and being corrected. Do not show a run that succeeds first time; nobody believes it.
