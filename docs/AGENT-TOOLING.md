# Agent tooling

What is wired in, what is optional, and what we checked before recommending it.

## Wired in (no action needed)

### `AGENTS.md` + `CLAUDE.md`
Architecture, commands, conventions, and the failure modes specific to this stack.
Claude Code and Codex both read `AGENTS.md`; `CLAUDE.md` points at it. **Keep it current** —
it is the cheapest, highest-leverage agent investment in the repo. When you change the
architecture, change this file in the same commit.

### Supabase MCP server (`.mcp.json`)
```json
{ "mcpServers": { "supabase": { "type": "http", "url": "http://localhost:54321/mcp" } } }
```

The local Supabase stack serves its own MCP endpoint, so this needs **no token and no
network** — it comes up with `supabase start`. An agent can then read your live schema and
run queries while building, instead of guessing column names.

In Claude Code, approve the server when prompted (or `/mcp` to check status). If the stack
is not running, the server is simply unavailable — nothing breaks.

**Hosted project instead of local?** Point it at `https://mcp.supabase.com/mcp`, optionally
with `?project_ref=<id>&read_only=true`. That endpoint uses a browser OAuth flow, so it
needs an interactive session to authorise once.

### Lovable MCP server (`.mcp.json`)
```json
{ "mcpServers": { "lovable": { "type": "http", "url": "https://mcp.lovable.dev" } } }
```

Lovable exposes itself as an MCP server, so an agent here can drive your Lovable projects:
create and deploy projects, send messages to the Lovable agent, read files and diffs, and
query its cloud database. Equivalent CLI one-liner:

```bash
claude mcp add --transport http lovable https://mcp.lovable.dev
```

**Auth:** the first Lovable tool call opens a browser to sign in — Claude Code, Claude, and
ChatGPT need only the URL. Available on all Lovable plans. Check it with `/mcp`.

Note the direction: this lets *this repo's agent* control Lovable. It does **not** give
Lovable access to this repo. If you want the reverse — the Lovable agent using your tools —
add a custom connector on the Lovable side instead.

**Delete the entry if you are not using Lovable.** An unauthenticated server just adds a
prompt and some noise to every session.

---

## Optional: `ui-ux-pro-max`

Design intelligence for coding agents — a searchable local database of UI styles, colour
palettes, font pairings, and per-stack (React / Tailwind / shadcn) design rules. Worth it
when UI polish is a judged differentiator.

```
/plugin marketplace add nextlevelbuilder/ui-ux-pro-max-skill
/plugin install ui-ux-pro-max@ui-ux-pro-max-skill
```

Or via CLI: `npm i -g ui-ux-pro-max-cli && uipro init --ai claude`

**What we checked** (clone inspected on 2026-08-20):
- MIT licensed. Active — last commit 2026-08-19.
- Ships 7 skills (`design`, `design-system`, `ui-styling`, `brand`, `banner-design`,
  `slides`, `ui-ux-pro-max`), ~11 MB, mostly CSV data + Markdown references.
- Scripts are Python 3 / Node with no added dependencies.
- ⚠️ The README claims the scripts "make no network calls". That is **not quite true** —
  `design-system/scripts/fetch-background.py` downloads stock photos from hardcoded
  `images.pexels.com` URLs. Benign, and only if you invoke that script, but the claim is
  wrong. Nothing else reaches the network; no telemetry, no credential access.
- `brand/scripts/sync-brand-to-tokens.cjs` shells out via `child_process.execFileSync` to
  sync design tokens. Read it before running it if that matters to you.

We install it as a plugin rather than vendoring 11 MB into the repo — keeps the diff clean
and lets teams skip it.

---

## Optional: `codegraph`

Pre-indexed semantic code graph exposed to agents over MCP. Instead of grepping file by
file, the agent queries a graph of symbols and their relationships. Most useful once the
codebase is big enough that navigation costs real time — or when a code-graph UI is itself
on-theme for your track.

```bash
npm i -g @colbymchenry/codegraph   # or the install.sh / install.ps1 one-liner
codegraph install                  # wires the MCP server
codegraph init                     # index this repo
```

**What we checked** (clone inspected on 2026-08-20):
- MIT licensed. Active — last commit 2026-08-07.
- Fully local: SQLite index, no API keys, no external services, no telemetry endpoints.
- No `postinstall`/`preinstall` hooks in `package.json`.
- Runs a **background daemon** that watches the filesystem and re-indexes on change.
  `CODEGRAPH_NO_DAEMON=1` disables shared-server mode.
- Bundles its own Node runtime and a Rust parsing kernel — the repo is ~154 MB. The
  published package is smaller, but expect a heavier install than a normal npm CLI.
- `.codegraph/` is gitignored.

**Cost/benefit for a 24-hour build:** the install is the heaviest thing in this document,
and a scaffold this small does not need graph navigation on day one. Add it if you have
slack, or if your track rewards it.

---

## Priority

If time is short, this is the order that actually pays off:

1. **`AGENTS.md` kept accurate** — free, compounding, helps every agent turn.
2. **Supabase MCP** — already wired, zero cost, removes schema guesswork.
3. **`ui-ux-pro-max`** — one command, real payoff when UI is judged.
4. **`codegraph`** — only with slack time.

Do not spend build hours integrating tooling. The scaffold exists so you write product code.
