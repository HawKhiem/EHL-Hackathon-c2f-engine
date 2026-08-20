# Task runner. Install: https://github.com/casey/just
# Everything here also works as a plain command - see README.md.

_default:
    @just --list

# Install + boot the whole stack
dev:
    ./setup.sh

# Install dependencies without booting
install:
    ./setup.sh --install-only

# Frontend only
web:
    cd frontend && npm run dev

# Backend only
api:
    cd backend && uv run uvicorn app.main:app --reload

# ---------------------------------------------------------------- database

# Wipe + re-migrate + re-seed local Postgres
db-reset:
    supabase db reset

# Local stack URLs and keys
db-status:
    supabase status

# New migration file
migration name:
    supabase migration new {{name}}

# ------------------------------------------------------------ dependencies

# Add a Python dependency (updates pyproject.toml + uv.lock)
add package:
    cd backend && uv add {{package}}

# Re-lock backend deps after editing pyproject.toml by hand
lock:
    cd backend && uv lock

# ----------------------------------------------------------------- quality

# Run tests
test:
    cd frontend && npm run test
    cd backend && uv run pytest -q

# Lint both sides
lint:
    cd frontend && npm run lint
    cd backend && uv run ruff check .

# Auto-fix and format everything
fix:
    cd frontend && npm run lint:fix && npm run format
    cd backend && uv run ruff check --fix . && uv run ruff format .

# Everything CI runs, in the same order. Run this before you push.
check:
    cd frontend && npm run lint && npm run format:check && npm run build && npm run test
    cd backend && uv run ruff check . && uv run ruff format --check . && uv run pytest -q
