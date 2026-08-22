#!/usr/bin/env bash
# Fetch the decryption key for a game and unzip its case as fast as possible.
# Usage:  ./get_case.sh [GAME_ID]        (default 0 = permanent test game)
#         TEAM_API_KEY must be set in env or in ./.env
set -euo pipefail
cd "$(dirname "$0")"

GAME_ID="${1:-0}"
BASE_URL="https://c2f.public.quantco.cloud"

# Load key from .env if not in env
if [[ -z "${TEAM_API_KEY:-}" && -f .env ]]; then
  TEAM_API_KEY="$(grep -E '^TEAM_API_KEY=' .env | cut -d= -f2- | tr -d '"'"'" )"
fi
[[ -n "${TEAM_API_KEY:-}" ]] || { echo "TEAM_API_KEY not set (env or .env)"; exit 1; }

# Locate 7z. c2f.submit resolves one in python and passes it in SEVENZ, because shell
# discovery here is not reliable on Windows: `pixi run` puts a 7z.exe on PATH that MSYS bash
# sometimes refuses ("cannot execute binary file"), and `command -v` sees a different PATH
# depending on whether bash was spawned by pixi or by python.exe. Round 34 was lost to that.
# The fallback below still works for a human running ./get_case.sh by hand.
if [[ -n "${SEVENZ:-}" ]]; then
  :
else
  SEVENZ="$(command -v 7z || command -v 7zz || command -v 7za || true)"
  for _c in .pixi/envs/default/bin/7z.exe .pixi/envs/default/bin/7z             .pixi/envs/default/Library/bin/7z.exe; do
    [[ -n "$SEVENZ" ]] || { [[ -f "$_c" ]] && SEVENZ="$_c"; } || true
  done
fi
[[ -n "$SEVENZ" ]] || { echo "7z not found. Run: pixi install   (or brew install p7zip)"; exit 1; }

CASE="$(printf 'case_%02d' "$GAME_ID")"
ZIP="cases/$CASE.zip"
OUT="cases/$CASE"
[[ -f "$ZIP" ]] || { echo "missing $ZIP"; exit 1; }

# Poll for the key (it 4xx's until start_time); retry quickly until it appears.
# A failed curl is a RETRY, never the end of the run: under `set -e` an unguarded
# command substitution would abort the whole script on one DNS or connection blip,
# and `-s` would swallow the reason (empty stdout, empty stderr, exit 6). That cost
# us game 25. The `|| true` and the timeouts below are what make the loop a loop.
# Progress goes to stderr so stdout stays parseable by the caller.
KEY_WAIT_S="${KEY_WAIT_S:-120}"
t0=$(date +%s)
while :; do
  RESP="$(curl -sS --connect-timeout 2 --max-time 5 -w '\n%{http_code}' \
            -H "X-API-Key: $TEAM_API_KEY" "$BASE_URL/api/games/$GAME_ID/key" 2>/dev/null || true)"
  CODE="${RESP##*$'\n'}"; BODY="${RESP%$'\n'*}"
  if [[ "$CODE" == "200" ]]; then break; fi
  if (( $(date +%s) - t0 > KEY_WAIT_S )); then
    echo "gave up after ${KEY_WAIT_S}s: code='${CODE:-none}' body='${BODY:0:200}'"
    exit 1
  fi
  printf '\r  waiting for key... (%s)' "${CODE:-conn}" >&2; sleep 0.3
done
KEY="$(printf '%s' "$BODY" | sed -E 's/.*"decryption_key" *: *"([^"]*)".*/\1/')"
echo; echo "KEY: $KEY"

rm -rf "$OUT"
"$SEVENZ" x -y -p"$KEY" -o"$OUT" "$ZIP" >/dev/null
echo "extracted -> $OUT/"; ls -1 "$OUT"
