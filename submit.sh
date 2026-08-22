#!/usr/bin/env bash
# Submit prices for a game.
# Usage:  ./submit.sh GAME_ID 'INDEX:CHARGE:LIMIT' ['INDEX:CHARGE:LIMIT' ...]
#    or:  ./submit.sh GAME_ID path/to/submissions.json
# Example: ./submit.sh 0 1:410:430 2:120:150
set -euo pipefail
cd "$(dirname "$0")"
GAME_ID="${1:?usage: ./submit.sh GAME_ID 'IDX:A:B' ... | file.json}"; shift
BASE_URL="https://c2f.public.quantco.cloud"

if [[ -z "${TEAM_API_KEY:-}" && -f .env ]]; then
  TEAM_API_KEY="$(grep -E '^TEAM_API_KEY=' .env | cut -d= -f2- | tr -d '"'"'" )"
fi
[[ -n "${TEAM_API_KEY:-}" ]] || { echo "TEAM_API_KEY not set (env or .env)"; exit 1; }

if [[ $# -eq 1 && -f "$1" ]]; then
  BODY="$(cat "$1")"
else
  items=()
  for spec in "$@"; do
    IFS=: read -r idx a b <<<"$spec"
    items+=("{\"index\":$idx,\"charge_price\":$a,\"acceptance_limit\":$b}")
  done
  BODY="[$(IFS=,; echo "${items[*]}")]"
fi

echo "PUT $BASE_URL/api/games/$GAME_ID/submissions"
echo "$BODY"
curl -s -w '\nHTTP %{http_code}\n' -X PUT \
  -H "X-API-Key: $TEAM_API_KEY" -H "Content-Type: application/json" \
  -d "$BODY" "$BASE_URL/api/games/$GAME_ID/submissions"
