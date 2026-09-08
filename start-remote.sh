#!/bin/bash
# Expose llama-server via Cloudflare tunnel for remote access
# Usage: ./start-remote.sh
#
# Run this on your PC (where llama-server is running).
# Then on your MacBook:
#
#   ANTHROPIC_BASE_URL=https://<tunnel-url> \
#   ANTHROPIC_AUTH_TOKEN=local \
#   claude --model qwen3.6:35b
#
# NOTE (updated 2026-09): llama-server now serves the Anthropic Messages API
# natively at /v1/messages (llama.cpp PR #17570), so tunnelling bare llama-server
# DOES work with Claude Code. Start it with --jinja for tool use. Ollama 0.14+
# and a LiteLLM proxy remain valid targets too.
#
# THIS SCRIPT PUBLISHES YOUR SERVER TO THE PUBLIC INTERNET via Cloudflare.
# If the client is just another machine on your own network (e.g. a MacBook),
# use the LAN-only path instead -- no tunnel, lower latency, far smaller
# exposure: serve-lan.ps1 on the server + ./connect-lan-mac.sh on the client.
# See "LAN-only access" in README.md.
#

set -e

PORT=8080
TUNNEL_LOG="/tmp/cloudflared-remote.log"

# Check llama-server is running
if ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
  echo "llama-server is not running on port $PORT"
  echo "Run ./start-server.sh first in another terminal."
  exit 1
fi

# Kill existing tunnel
pkill -f "cloudflared.*$PORT" 2>/dev/null && sleep 1 || true

echo "Starting Cloudflare tunnel to llama-server..."
nohup cloudflared tunnel --url "http://localhost:$PORT" --protocol http2 > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

# Wait for tunnel URL
TUNNEL_URL=""
for i in $(seq 1 20); do
  TUNNEL_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | grep -v api | head -1)
  if [ -n "$TUNNEL_URL" ]; then
    break
  fi
  sleep 1
done

if [ -z "$TUNNEL_URL" ]; then
  echo "Failed to get tunnel URL. Check $TUNNEL_LOG"
  exit 1
fi

echo ""
echo "============================================"
echo "  Remote LLM Access Ready"
echo "============================================"
echo ""
echo "  Tunnel URL: $TUNNEL_URL"
echo ""
echo "  On your MacBook, run:"
echo ""
echo "    ANTHROPIC_BASE_URL=$TUNNEL_URL \\"
echo "    ANTHROPIC_AUTH_TOKEN=local \\"
echo "    claude --model qwen3.6:35b"
echo ""
echo "  Tunnel PID: $TUNNEL_PID (log: $TUNNEL_LOG)"
echo "  Press Ctrl+C to stop."
echo "============================================"

trap "kill $TUNNEL_PID 2>/dev/null; echo 'Stopped.'; exit 0" INT TERM
wait
