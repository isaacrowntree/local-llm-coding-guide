#!/bin/bash
# Launch Claude Code connected to local Qwen
# Usage: ./start-claude-local.sh [optional-directory]
#
# Opens Claude Code using your local llama-server as the backend.
# Auto-detects the running model. Make sure start-server.sh or start-server-mac.sh is running first.

set -e

PORT=8080
DIR="${1:-.}"

# Check llama-server is running
if ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
  echo "llama-server is not running on port $PORT"
  echo "Run ./start-server.sh or ./start-server-mac.sh first in another terminal."
  exit 1
fi

# Auto-detect model name from running server
MODEL_ID=$(curl -s "http://localhost:$PORT/v1/models" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
MODEL="openai/${MODEL_ID:-qwen}"

echo "Starting Claude Code with local model..."
echo "Model: $MODEL"
echo "Endpoint: http://localhost:$PORT"
echo ""

cd "$DIR"
ANTHROPIC_BASE_URL="http://localhost:$PORT" \
ANTHROPIC_AUTH_TOKEN="local" \
claude --model "$MODEL"
