#!/bin/bash
# Start LiteLLM proxy + Cloudflare tunnel for Cursor
# Usage: ./start-cursor-local.sh
#
# Make sure start-server.sh is running first.
# After this script starts, configure Cursor:
#   1. Settings -> Models -> + Add Custom Model -> "deepseek-r1-0528"
#   2. Override OpenAI Base URL -> the tunnel URL printed below + /v1
#   3. API Key -> sk-1234
#   4. Use Chat or Cmd+K mode (Agent mode does NOT support custom endpoints)

set -e

PORT=8080
LITELLM_PORT=4000
CONFIG_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$CONFIG_DIR/litellm-config.yaml"
LITELLM_LOG="/tmp/litellm.log"
TUNNEL_LOG="/tmp/cloudflared.log"

# Check llama-server is running
if ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
  echo "llama-server is not running on port $PORT"
  echo "Run ./start-server.sh first in another terminal."
  exit 1
fi

# Create LiteLLM config
cat > "$CONFIG" << 'EOF'
model_list:
  - model_name: deepseek-r1-0528
    litellm_params:
      model: openai/deepseek-r1-0528
      api_base: http://localhost:8080/v1
      api_key: local
EOF

# Kill existing processes
pkill -f "litellm" 2>/dev/null && sleep 1 || true
pkill -f "cloudflared" 2>/dev/null && sleep 1 || true

# Start LiteLLM proxy
echo "Starting LiteLLM proxy on port $LITELLM_PORT..."
nohup litellm --config "$CONFIG" --port $LITELLM_PORT --host 0.0.0.0 > "$LITELLM_LOG" 2>&1 &
LITELLM_PID=$!

# Wait for LiteLLM to start
echo "Waiting for LiteLLM..."
for i in $(seq 1 15); do
  if curl -s "http://localhost:$LITELLM_PORT/v1/models" > /dev/null 2>&1; then
    echo "LiteLLM ready."
    break
  fi
  sleep 1
done

# Start Cloudflare tunnel
echo "Starting Cloudflare tunnel..."
nohup cloudflared tunnel --url "http://localhost:$LITELLM_PORT" --protocol http2 > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

# Wait for tunnel URL
echo "Waiting for tunnel URL..."
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
echo "  Cursor Local LLM Setup Ready"
echo "============================================"
echo ""
echo "  Tunnel URL: $TUNNEL_URL"
echo ""
echo "  Cursor Settings:"
echo "    Model:    deepseek-r1-0528"
echo "    Base URL: $TUNNEL_URL/v1"
echo "    API Key:  sk-1234"
echo ""
echo "  NOTE: Only works in Chat and Cmd+K mode."
echo "        Agent mode does NOT support custom endpoints."
echo ""
echo "  LiteLLM PID: $LITELLM_PID (log: $LITELLM_LOG)"
echo "  Tunnel PID:  $TUNNEL_PID (log: $TUNNEL_LOG)"
echo ""
echo "  Press Ctrl+C to stop."
echo "============================================"

# Wait and keep running
trap "kill $LITELLM_PID $TUNNEL_PID 2>/dev/null; echo 'Stopped.'; exit 0" INT TERM
wait
