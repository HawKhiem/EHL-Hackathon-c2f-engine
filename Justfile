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

# Wipe + re-migrate + re-seed local Postgres
db-reset:
    supabase db reset

# Local stack URLs and keys
db-status:
    supabase status

# New migration file
migration name:
    supabase migration new {{name}}

# Add a Python dependency (updates pyproject.toml + uv.lock)
add package:
    cd backend && uv add {{package}}

# Re-lock backend deps after editing pyproject.toml by hand
lock:
    cd backend && uv lock

# Typecheck frontend, import backend, lint backend
check:
    cd frontend && npm run build
    cd backend && uv run python -c "from app.main import app; print('backend ok')"
    cd backend && uv run ruff check .
