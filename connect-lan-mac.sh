#!/bin/bash
# Run Claude Code on this Mac against llama-server on another machine on your LAN.
#
#   ./connect-lan-mac.sh                       # just runs -- no arguments needed
#   ./connect-lan-mac.sh -p "fix the test"     # extra args pass through to claude
#
# Config is read from the first of:
#   1. environment  (LLAMA_HOST / LLAMA_KEY / LLAMA_PORT / LLAMA_MODEL)
#   2. ./.llama-lan.conf
#   3. ~/.llama-lan.conf
#
# Create ~/.llama-lan.conf once (chmod 600 -- it holds the API key):
#   LLAMA_HOST=192.168.5.195
#   LLAMA_KEY=llk_...
#
# The server side is serve-lan.ps1 on the Windows box. llama-server speaks the
# Anthropic Messages API natively, so there is no proxy in this path.
set -euo pipefail

for f in "./.llama-lan.conf" "$HOME/.llama-lan.conf"; do
  [ -f "$f" ] && . "$f" && break
done

HOST="${LLAMA_HOST:-}"
PORT="${LLAMA_PORT:-8080}"
MODEL="${LLAMA_MODEL:-qwen3.6-local}"
KEY="${LLAMA_KEY:-}"

if [ -z "$HOST" ]; then
  echo "No LLAMA_HOST configured." >&2
  echo "Create ~/.llama-lan.conf with LLAMA_HOST= and LLAMA_KEY= (see header)." >&2
  exit 1
fi
[ -n "$KEY" ] || { read -rsp "API key for http://$HOST:$PORT : " KEY; echo; }

BASE="http://$HOST:$PORT"

# Preflight: fail with a useful message instead of letting Claude Code retry blindly.
code=$(curl -s -o /dev/null -w '%{http_code}' -m 8 -H "Authorization: Bearer $KEY" "$BASE/v1/models" || echo 000)
case "$code" in
  200) : ;;
  401|403) echo "ERROR: server reachable but rejected the API key (check LLAMA_KEY)." >&2; exit 1 ;;
  000) echo "ERROR: cannot reach $BASE" >&2
       echo "  - serve-lan.ps1 running on the server?" >&2
       echo "  - firewall rule installed (setup-lan-firewall.ps1, as admin)?" >&2
       echo "  - both machines on the same subnet?" >&2
       exit 1 ;;
  *)   echo "ERROR: unexpected HTTP $code from $BASE/v1/models" >&2; exit 1 ;;
esac

command -v claude >/dev/null || { echo "ERROR: 'claude' not on PATH." >&2; exit 1; }
echo "Claude Code -> $BASE  ($MODEL)"

# ANTHROPIC_API_KEY is unset so it cannot outrank the bearer token (auth precedence).
# WebSearch/WebFetch are Anthropic server-side tools; llama-server 400s on them.
exec env -u ANTHROPIC_API_KEY \
  ANTHROPIC_BASE_URL="$BASE" \
  ANTHROPIC_AUTH_TOKEN="$KEY" \
  ANTHROPIC_MODEL="$MODEL" \
  ANTHROPIC_SMALL_FAST_MODEL="$MODEL" \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
  claude --model "$MODEL" --disallowedTools WebSearch WebFetch "$@"
