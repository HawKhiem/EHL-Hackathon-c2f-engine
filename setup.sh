#!/usr/bin/env bash
# ============================================================================
#  One-command setup + boot.
#
#    ./setup.sh                 install everything, then run the whole stack
#    ./setup.sh --install-only  install, do not boot
#    ./setup.sh --reset-db      wipe + re-migrate + re-seed the local DB
#    ./setup.sh --no-db         skip Supabase entirely (no Docker needed)
#
#  Idempotent: safe to re-run. Fails loud on the first real problem.
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"

INSTALL_ONLY=0
RESET_DB=0
USE_DB=1
for arg in "$@"; do
  case "$arg" in
    --install-only) INSTALL_ONLY=1 ;;
    --reset-db)     RESET_DB=1 ;;
    --no-db)        USE_DB=0 ;;
    -h|--help)      sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown flag: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# ----------------------------------------------------------------- output
if [ -t 1 ]; then
  BOLD=$(printf '\033[1m'); DIM=$(printf '\033[2m')
  RED=$(printf '\033[31m'); GREEN=$(printf '\033[32m')
  YELLOW=$(printf '\033[33m'); CYAN=$(printf '\033[36m')
  OFF=$(printf '\033[0m')
else
  BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; CYAN=""; OFF=""
fi

step() { printf "\n%s==>%s %s%s%s\n" "$CYAN" "$OFF" "$BOLD" "$1" "$OFF"; }
ok()   { printf "  %s[ok]%s %s\n" "$GREEN" "$OFF" "$1"; }
bad()  { printf "  %s[--]%s %s\n" "$RED" "$OFF" "$1"; }
warn() { printf "  %s[!!]%s %s\n" "$YELLOW" "$OFF" "$1"; }
die()  { printf "\n%serror:%s %s\n" "$RED" "$OFF" "$1" >&2; exit 1; }

# ----------------------------------------------------------- 1. prereqs
step "Checking prerequisites"

MISSING=0

need() {
  # need <cmd> <label> <install-hint>
  if command -v "$1" >/dev/null 2>&1; then
    ok "$2 ${DIM}$("$1" --version 2>&1 | head -1)${OFF}"
  else
    bad "$2 not found - $3"
    MISSING=1
  fi
}

version_at_least() {
  # version_at_least <have> <want>
  [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]
}

need node "node" "install Node >= 20 from https://nodejs.org (or use nvm)"
need npm  "npm"  "ships with Node"
need git  "git"  "https://git-scm.com/downloads"

# uv manages the Python toolchain, the virtualenv, and the lockfile.
# It will download a suitable CPython itself, so Python is NOT a separate
# prerequisite. Install: https://docs.astral.sh/uv/getting-started/installation
need uv "uv" "curl -LsSf https://astral.sh/uv/install.sh | sh  (Windows: irm https://astral.sh/uv/install.ps1 | iex)"

if [ "$USE_DB" -eq 1 ]; then
  need docker   "docker"   "https://docs.docker.com/get-docker (must be RUNNING)"
  need supabase "supabase" "npm i -g supabase - or https://supabase.com/docs/guides/cli"
fi

[ "$MISSING" -eq 0 ] || die "install the missing prerequisites above, then re-run ./setup.sh"

NODE_V="$(node --version | tr -d 'v')"
version_at_least "$NODE_V" "20.0.0" || die "node $NODE_V is too old; need >= 20"

if [ "$USE_DB" -eq 1 ] && ! docker info >/dev/null 2>&1; then
  die "Docker is installed but not running. Start Docker Desktop and re-run."
fi

# ------------------------------------------------------------- 2. .env
step "Configuring .env"
if [ -f .env ]; then
  ok ".env already exists (left untouched)"
else
  cp .env.example .env
  ok "created .env from .env.example"
fi

set_env() {
  # set_env <KEY> <value> - replace in place, or append if absent.
  # awk instead of sed -i because BSD and GNU sed disagree on the -i syntax.
  key="$1"
  value="$2"
  if grep -q "^${key}=" .env; then
    awk -v k="$key" -v v="$value" '
      { if (index($0, k "=") == 1) print k "=" v; else print }
    ' .env > .env.tmp && mv .env.tmp .env
  else
    printf '%s=%s\n' "$key" "$value" >> .env
  fi
}

read_env() {
  # read_env <KEY> - echo the value from .env, empty if unset
  sed -n "s/^$1=//p" .env | head -1
}

# ---------------------------------------------------------- 3. frontend
step "Installing frontend dependencies"
(cd frontend && npm install --no-fund --no-audit) || die "npm install failed"
ok "frontend/node_modules ready"

# ----------------------------------------------------------- 4. backend
step "Installing backend dependencies"
# `uv sync` creates backend/.venv, provisions Python if needed, and installs
# exactly what uv.lock pins. --frozen keeps it honest: if pyproject.toml and
# uv.lock disagree it fails instead of silently re-resolving.
if [ -f backend/uv.lock ]; then
  (cd backend && uv sync --frozen) \
    || die "uv sync failed. If you changed pyproject.toml, run: cd backend && uv lock"
else
  (cd backend && uv sync) || die "uv sync failed"
fi
ok "backend deps installed ${DIM}(backend/.venv)${OFF}"

# ---------------------------------------------------------- 5. supabase
if [ "$USE_DB" -eq 1 ]; then
  step "Starting Supabase (local Postgres + Studio)"

  if supabase status >/dev/null 2>&1; then
    ok "Supabase already running"
  else
    if ! supabase start; then
      # By far the most common cause: another local Supabase project already
      # owns the default ports. `supabase status` is per-project, so a stack
      # from a different repo does not show up as "already running".
      printf "\n"
      warn "If the error above mentions a port already allocated, another local"
      warn "Supabase project owns this branch's ports (54621-54627). Either:"
      warn "  1. stop it:  cd <other-project> && supabase stop"
      warn "  2. or give THIS project its own ports in supabase/config.toml"
      warn "     (api.port, db.port, studio.port, ...) and re-run."
      printf "\n"
      docker ps --filter "publish=54622" --format '  holding 54322: {{.Names}}' 2>/dev/null || true
      die "supabase start failed"
    fi
    ok "Supabase started"
  fi

  if [ "$RESET_DB" -eq 1 ]; then
    supabase db reset || die "supabase db reset failed"
    ok "database reset, migrations + seed applied"
  else
    supabase migration up || die "could not apply migrations - try ./setup.sh --reset-db"
    ok "migrations applied"
  fi

  step "Writing Supabase keys into .env"
  STATUS="$(supabase status -o env 2>/dev/null || true)"
  get_status() {
    printf '%s\n' "$STATUS" | sed -n "s/^$1=//p" | head -1 | tr -d '"'
  }

  # Prefer the current publishable/secret keys. The CLI also still prints the
  # legacy ANON_KEY / SERVICE_ROLE_KEY JWTs, which Supabase deprecates by the
  # end of 2026 - we deliberately do not write those into .env.
  API_URL="$(get_status API_URL)"
  PUBLISHABLE="$(get_status PUBLISHABLE_KEY)"
  SECRET="$(get_status SECRET_KEY)"
  [ -n "$API_URL" ] || API_URL="http://127.0.0.1:54621"

  if [ -n "$PUBLISHABLE" ] && [ -n "$SECRET" ]; then
    set_env SUPABASE_URL "$API_URL"
    set_env SUPABASE_PUBLISHABLE_KEY "$PUBLISHABLE"
    set_env SUPABASE_SECRET_KEY "$SECRET"
    set_env VITE_SUPABASE_URL "$API_URL"
    set_env VITE_SUPABASE_PUBLISHABLE_KEY "$PUBLISHABLE"
    ok "keys written to .env ${DIM}(publishable + secret)${OFF}"
  else
    warn "could not parse publishable/secret keys from supabase status"
    warn "your Supabase CLI may predate them - run: supabase status -o env"
    warn "then copy the values into .env by hand"
  fi
else
  warn "skipping Supabase (--no-db)"
fi

# ------------------------------------------------------- 6. sanity check
step "Verifying the backend imports"
(cd backend && uv run --frozen python -c \
  "from app.main import app; print('  routes registered:', len(app.routes))") \
  || die "backend failed to import - see the traceback above"

if [ -z "$(read_env ANTHROPIC_API_KEY)" ]; then
  warn "ANTHROPIC_API_KEY is empty in .env - /llm/* will fail until you set it"
fi

# ------------------------------------------------------------- 7. boot
if [ "$INSTALL_ONLY" -eq 1 ]; then
  step "Done (install only)"
  printf "  Run %s./setup.sh%s to boot the stack.\n\n" "$BOLD" "$OFF"
  exit 0
fi

step "Booting"

VITE_PORT="$(sed -n 's/.*port: \([0-9]\{4\}\),.*/\1/p' frontend/vite.config.ts | head -1)"
[ -n "$VITE_PORT" ] || VITE_PORT=5173
BACKEND_PORT="$(read_env BACKEND_PORT)"
[ -n "$BACKEND_PORT" ] || BACKEND_PORT=8000

BACKEND_PID=""
cleanup() {
  trap - INT TERM EXIT
  if [ -n "$BACKEND_PID" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

(cd backend && exec uv run --frozen uvicorn app.main:app --reload \
  --host 127.0.0.1 --port "$BACKEND_PORT") &
BACKEND_PID=$!

# Wait for the backend to answer before starting Vite, so a crash on boot
# is obvious instead of being buried under Vite's output.
i=0
while [ "$i" -lt 20 ]; do
  kill -0 "$BACKEND_PID" 2>/dev/null || die "backend exited during startup (see traceback above)"
  if command -v curl >/dev/null 2>&1; then
    if curl -fsS "http://127.0.0.1:$BACKEND_PORT/health" >/dev/null 2>&1; then
      break
    fi
  else
    sleep 2
    break
  fi
  i=$((i + 1))
  sleep 1
done

printf "\n  %sStack is up%s\n" "$BOLD" "$OFF"
printf "  %sapp%s              http://localhost:%s\n" "$GREEN" "$OFF" "$VITE_PORT"
printf "  %sapi docs%s         http://127.0.0.1:%s/docs\n" "$GREEN" "$OFF" "$BACKEND_PORT"
printf "  %ssupabase studio%s  http://127.0.0.1:54623\n" "$GREEN" "$OFF"
printf "\n  %sCtrl-C stops both servers.%s\n\n" "$DIM" "$OFF"

(cd frontend && exec npm run dev)
