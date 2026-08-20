# Hackathon Scaffold

Vite + React + TypeScript + Tailwind + shadcn/ui · FastAPI · Supabase · provider-agnostic LLM.

Everything below the product layer is done and verified. You should be writing feature code
within ten minutes of cloning.

## 60-second quickstart

```bash
git clone <your-repo-url> && cd <repo>
./setup.sh
```

That installs frontend + backend deps, boots local Supabase, applies migrations and seed
data, writes the Supabase keys into `.env` for you, and starts both dev servers.

| URL | What |
|---|---|
| http://localhost:5173 | The app |
| http://127.0.0.1:8000/docs | API docs (OpenAPI, interactive) |
| http://127.0.0.1:54323 | Supabase Studio |

**One manual step:** paste your `ANTHROPIC_API_KEY` into `.env`. Everything else is
filled in automatically. The four pills at the top of the app go green when the stack
is fully wired.

### Prerequisites

Node ≥ 20 · [uv](https://docs.astral.sh/uv/getting-started/installation/) · Docker (running) · [Supabase CLI](https://supabase.com/docs/guides/cli)

```bash
# uv — macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# uv — Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

**You do not need to install Python.** uv provisions the right CPython itself, creates
`backend/.venv`, and installs from the committed lockfile.

`setup.sh` checks every prerequisite and prints an install hint for anything missing.
No Docker? Run `./setup.sh --no-db` to work without the database.

## Common commands

| Command | What |
|---|---|
| `./setup.sh` | Install + boot everything |
| `./setup.sh --install-only` | Install, don't boot |
| `./setup.sh --reset-db` | Wipe + re-migrate + re-seed local Postgres |
| `./setup.sh --no-db` | Skip Supabase entirely |
| `cd frontend && npm run dev` | Frontend only |
| `cd frontend && npm run build` | Typecheck + production build |
| `cd backend && uv run uvicorn app.main:app --reload` | Backend only |
| `cd backend && uv add <pkg>` | Add a Python dependency |
| `supabase db reset` | Reapply migrations + seed |
| `supabase status` | Local stack URLs and keys |
| `npx shadcn@latest add dialog` | Add a shadcn component |
| `just check` | Everything CI runs (lint, format, types, tests, build) |
| `just fix` | Auto-fix lint + formatting on both sides |
| `just test` | Tests only |

A `Justfile` wraps these if you have [`just`](https://github.com/casey/just) installed;
`just` on its own lists every target.

## Quality gates

| | |
|---|---|
| Backend | `ruff` (lint + format), `pytest` — 21 tests |
| Frontend | `eslint`, `prettier`, `tsc`, `vitest` — 9 tests |
| CI | `.github/workflows/ci.yml` — runs all of the above on every push and PR |

`just check` runs exactly what CI runs, so a green local check means a green pipeline.
Tests never touch a real LLM or a real network: the backend swaps in a `FakeProvider`,
the frontend stubs `fetch`.

CI is **verification only** — it does not deploy. Roughly two minutes, with dependency
caching and superseded runs cancelled automatically.

## Layout

```
frontend/          Vite + React + TS + Tailwind v4 + shadcn/ui
  src/lib/api.ts   typed client for the backend  <- add endpoints here
  src/index.css    every design token            <- restyle from here
backend/           FastAPI (Python + deps managed by uv)
  pyproject.toml   Python dependencies      <- `uv add <pkg>`
  uv.lock          committed, exact pins
  app/routers/     one file per feature area, auto-registered
  app/llm/         provider-agnostic LLM wrapper
supabase/          migrations + seed
setup.sh           install + run everything
AGENTS.md          instructions for coding agents  <- read this
CHALLENGE.md       paste the real brief here at kick-off
.github/workflows  CI: lint, typecheck, test, build
```

## Working with coding agents

This repo is set up to be built with Claude Code / Codex:

- **`AGENTS.md`** — architecture, conventions, and the traps worth knowing. Both Claude Code
  and Codex read it. Keep it current; it is the highest-leverage file here.
- **`.mcp.json`** — wires two MCP servers:
  - **Supabase**, at the local stack's own endpoint (`http://localhost:54321/mcp`), so an
    agent can inspect your schema and run queries while it builds. No token needed — it
    comes up with `supabase start`.
  - **Lovable**, at `https://mcp.lovable.dev`. First use opens a browser to sign in.
    Delete the entry if your team is not using Lovable.
- **`docs/AGENT-TOOLING.md`** — MCP details plus two optional third-party skills (UI/UX
  design intelligence and a codebase graph), with vetting notes and one-line installs.

## Notes

- `.env` is gitignored. Document new keys in `.env.example`.
- Anything prefixed `VITE_` is compiled into the browser bundle — public by definition.
- Supabase keys: `sb_publishable_...` is browser-safe (RLS applies), `sb_secret_...` is
  backend-only (bypasses RLS). These replace the legacy anon/service_role JWTs.
- **Every new table needs `grant` statements as well as RLS policies** — see `AGENTS.md`.
  Without the grants you get `42501 permission denied` no matter what your policies say.
- The `notes` table and the `StatusBar` / `ChatPanel` components are scaffolding. Delete
  them once your real feature exists.
