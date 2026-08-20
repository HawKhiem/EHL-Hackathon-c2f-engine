# AGENTS.md

Instructions for coding agents (Claude Code, Codex, Cursor) working in this repo.
Humans: read `README.md` first, then this.

---

## What this is

A hackathon scaffold. The product code does not exist yet — you are here to write it.
The plumbing (auth, DB, LLM calls, dev servers, one-command setup) is already done and
**works**. Do not rebuild it.

| Layer | Stack | Lives in |
|---|---|---|
| Frontend | Vite + React 19 + TypeScript + Tailwind v4 + shadcn/ui | `frontend/` |
| Backend | FastAPI (Python 3.11+), async | `backend/` |
| Data / auth / storage / realtime | Supabase (local Postgres via Docker) | `supabase/` |
| LLM | Provider-agnostic wrapper, Anthropic by default | `backend/app/llm/` |

---

## Run it

```bash
./setup.sh                 # install everything + boot the whole stack
./setup.sh --install-only  # install, don't boot
./setup.sh --reset-db      # wipe + re-migrate + re-seed local Postgres
./setup.sh --no-db         # skip Supabase (no Docker required)
```

| URL | What |
|---|---|
| http://localhost:5173 | The app |
| http://127.0.0.1:8000/docs | FastAPI interactive docs (OpenAPI) |
| http://127.0.0.1:54323 | Supabase Studio (browse tables, run SQL) |

Individual servers, if you need them separately:

```bash
cd frontend && npm run dev
cd backend  && uv run uvicorn app.main:app --reload
supabase start | supabase stop | supabase db reset
```

**Python is managed entirely by [`uv`](https://docs.astral.sh/uv/).** Never activate a
venv, never call `pip`, never invoke `python` directly for backend code — prefix with
`uv run` and it uses `backend/.venv` automatically:

```bash
cd backend
uv run uvicorn app.main:app --reload   # run the server
uv run python -c "..."                 # run anything in the venv
uv run ruff check .                    # lint
uv add <package>                       # add a dep (updates pyproject.toml + uv.lock)
uv remove <package>                    # drop a dep
uv sync                                # match the venv to uv.lock
```

Dependencies live in `backend/pyproject.toml`. `backend/uv.lock` is **committed** so the
whole team resolves identical versions — commit it whenever it changes. There is no
`requirements.txt`. `setup.sh` runs `uv sync --frozen`, which fails loudly if
`pyproject.toml` and `uv.lock` have drifted apart; the fix is `uv lock`.

---

## Where things go

```
frontend/src/
  components/ui/     shadcn primitives. Regenerate with `npx shadcn@latest add <name>`.
  components/        your feature components
  lib/api.ts         THE typed backend client — add a method here, never fetch() in a component
  lib/supabase.ts    browser Supabase client (anon key, RLS applies)
  lib/utils.ts       cn() class merger
  index.css          all design tokens — restyle the app from here
  theme.css          per-project token overrides (loaded after index.css, wins)

backend/app/
  main.py            FastAPI app, CORS, router auto-discovery
  config.py          all env config, validated once (pydantic-settings)
  routers/           one file per feature area — AUTO-REGISTERED, see below
  llm/               provider-agnostic LLM wrapper — see below
  supabase_client.py server-side client (SERVICE ROLE — bypasses RLS)

supabase/
  migrations/        forward-only SQL. New change = new file, never edit an applied one.
  seed.sql           local dev data, re-applied on `supabase db reset`
```

---

## The LLM wrapper

Always go through the factory. Never import a provider directly, never construct an
SDK client in a router.

```python
from app.llm import get_llm

provider = get_llm()

result = await provider.complete(messages, system="You are…")   # one-shot
async for token in provider.stream(messages): ...               # streaming
```

Switching providers is an env change (`LLM_PROVIDER=anthropic|openai`), never a code change.

**Things that will bite you if you write Anthropic calls by hand:**

- Default model is `claude-opus-5`. Use the exact id — no date suffix.
- Thinking is `thinking={"type": "adaptive"}`. `budget_tokens` is **removed** on this
  model family and returns a 400.
- A request can be *declined*: HTTP **200** with `stop_reason == "refusal"`, sometimes with
  an empty `content` list. Check `stop_reason` before reading `content[0]` or you will
  crash on a success response. The wrapper already does this.
- Assistant-message prefill returns a 400. Use a system prompt or structured outputs.
- Long or high-`max_tokens` responses must stream, or you will hit HTTP timeouts.

New LLM feature? Add a method to the `LLMProvider` protocol in `llm/base.py`, then
implement it in each provider so the contract stays honest.

---

## Adding an endpoint

Routers are **auto-discovered**. Any module in `backend/app/routers/` that defines a
module-level `router = APIRouter(...)` is registered at startup — `main.py` needs no edit.
Each registration is logged, so check the boot output if an endpoint seems missing.

```python
# backend/app/routers/widgets.py
from fastapi import APIRouter

router = APIRouter(prefix="/widgets", tags=["widgets"])

@router.get("")
async def list_widgets() -> list[str]:
    return []
```

Then add the matching type + method to `frontend/src/lib/api.ts`.

---

## Database rules

### Every new table needs GRANTs *and* RLS policies

This is the single most common way to lose an hour here. A role needs **both** table
privileges and a policy that passes. Miss the grants and the API returns
`42501 permission denied for table x` however perfect your policies are.

It is not automatic: default privileges on `public` are owned by `supabase_admin`, but
migrations run as `postgres`, so tables you create inherit nothing.

```sql
-- required for every new table
grant select on public.thing to anon;
grant select, insert, update, delete on public.thing to authenticated;
grant all on public.thing to service_role;

alter table public.thing enable row level security;
create policy thing_select_own on public.thing
  for select to authenticated
  using ((select auth.uid()) = user_id);
```

### Keys and roles

| API key | DB role | Where | RLS |
|---|---|---|---|
| `sb_publishable_...` | `anon` | browser | applies |
| user's JWT | `authenticated` | browser | applies |
| `sb_secret_...` | `service_role` | backend only | **bypassed** |

These replaced the legacy `anon` / `service_role` JWT keys, which Supabase deprecates by
the end of 2026. The CLI still prints the old ones — do not use them.

### The rest

- **RLS on with no policy denies everything; RLS off is a data leak.** Enable it on every
  table the browser can reach, and write the policy in the same migration.
- Wrap auth calls in a subquery — `(select auth.uid()) = user_id` — so Postgres evaluates
  them once per statement instead of once per row.
- Migrations are forward-only. A schema change is a **new** file in `supabase/migrations/`.
  Editing an already-applied migration desyncs every teammate.
- `supabase_client.py` uses the secret key and **bypasses RLS**. Never return rows from it
  to a caller you have not authorised, and never send that key to the browser.
- The init migration ships an anon-readable policy (`notes_select_demo_rows`) so the
  scaffold shows data before login exists. **Delete it before you demo anything real.**

---

## Conventions

- **Types cross the boundary.** A new endpoint means a matching type + method in
  `frontend/src/lib/api.ts`. Check your shapes against `/docs`.
- **Config comes from `config.py`.** No `os.environ` reads scattered through routers.
- **Secrets stay server-side.** Vite inlines every `VITE_*` var into the browser bundle,
  so anything under `VITE_` is public. The anon key is fine; nothing else is.
- **Restyle via `index.css` tokens**, not per-component hex values — that is what makes a
  fast visual re-theme possible.
- Frontend: 2-space indent, named exports, `@/` import alias.
- Backend: `from __future__ import annotations`, type hints on public functions,
  `async def` for anything touching I/O. `ruff` config is in `backend/pyproject.toml`.

---

## Writing anything a human will read

Pitch copy, the project description, README text, UI microcopy, demo scripts — these are
**judged**. Write them like a person who built the thing:

- Lead with what it does and who it is for. No "In today's fast-paced world."
- Concrete over abstract: name the actual user, the actual number, the actual before/after.
- Cut hedges ("arguably", "we believe"), cut triples ("fast, scalable, and robust"),
  cut em-dash-heavy rhythm and "It's not just X, it's Y" constructions.
- Vary sentence length. Short sentences carry weight.
- Claim only what the demo actually does. Judges probe. Aspirational claims collapse.

---

## Do not touch

Unless the task is explicitly about them:

- `setup.sh` — it is verified working. Breaking it costs the whole team time.
- `.env` — gitignored, holds real keys. Edit `.env.example` to document a *new* key.
- `frontend/src/components/ui/*` — regenerate via the shadcn CLI instead of hand-editing.
- Applied files in `supabase/migrations/` — add a new migration.
- `.mcp.json` — shared agent tooling config.

## Delete when you outgrow them

These exist to prove the wiring and to be replaced:

- `frontend/src/components/StatusBar.tsx` — boot check
- `frontend/src/components/ChatPanel.tsx` — LLM streaming reference
- the `notes` table + its demo policy

---

## Before you say you are done

```bash
cd frontend && npm run build                              # typecheck + production build
cd backend  && uv run python -c "from app.main import app"  # backend imports cleanly
cd backend  && uv run ruff check .                        # lint
```

Then load the app and click the thing you changed. A green typecheck is not a working
feature.
