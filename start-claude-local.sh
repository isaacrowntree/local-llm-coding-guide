#!/bin/bash
# Launch Claude Code connected to local Qwen3.5-9B
# Usage: ./start-claude-local.sh [optional-directory]
#
# Opens Claude Code using your local llama-server as the backend.
# Make sure start-server.sh is running first.

set -e

PORT=8080
MODEL="openai/qwen-3.5-9b"
DIR="${1:-.}"

# Check llama-server is running
if ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
  echo "llama-server is not running on port $PORT"
  echo "Run ./start-server.sh first in another terminal."
  exit 1
fi

echo "Starting Claude Code with local Qwen3.5-9B..."
echo "Model: $MODEL"
echo "Endpoint: http://localhost:$PORT"
echo ""

cd "$DIR"
ANTHROPIC_BASE_URL="http://localhost:$PORT" \
ANTHROPIC_AUTH_TOKEN="local" \
claude --model "$MODEL"
