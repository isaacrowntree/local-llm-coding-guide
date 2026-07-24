#!/bin/bash
# Launch Claude Code connected to a local model
# Usage: ./start-claude-local.sh [optional-directory]
#
# Claude Code speaks ONLY the Anthropic Messages API. Ollama 0.14+ serves that
# natively (no proxy, no openai/ prefix), so this script prefers Ollama.
# llama-server / TabbyAPI / TRT are OpenAI-only and need a LiteLLM proxy — this
# script does not set one up; use Ollama for the zero-proxy path.

set -e

DIR="${1:-.}"

# Prefer Ollama (native Anthropic /v1/messages) on 11434
if curl -s "http://localhost:11434/api/tags" > /dev/null 2>&1; then
  MODEL=$(curl -s "http://localhost:11434/api/tags" | grep -o '"model":"[^"]*"' | head -1 | cut -d'"' -f4)
  MODEL="${MODEL:-qwen3.6:35b}"
  BASE_URL="http://localhost:11434"
  echo "Using Ollama (native Anthropic API) — no proxy needed"
elif curl -s "http://localhost:8080/v1/models" > /dev/null 2>&1; then
  echo "Found llama-server on :8080, but it is OpenAI-only — Claude Code cannot use it directly."
  echo "Either run Ollama (./start-ollama-mac.sh) for a native no-proxy path, or put a"
  echo "LiteLLM proxy in front of llama-server (see the Cursor/LiteLLM section in the README)."
  exit 1
else
  echo "No local model server found."
  echo "Start one first: ./start-ollama-mac.sh (recommended) or ./start-server-mac.sh + a LiteLLM proxy."
  exit 1
fi

echo "Starting Claude Code with local model..."
echo "Model: $MODEL"
echo "Endpoint: $BASE_URL"
echo ""

cd "$DIR"
ANTHROPIC_BASE_URL="$BASE_URL" \
ANTHROPIC_AUTH_TOKEN="local" \
ANTHROPIC_SMALL_FAST_MODEL="$MODEL" \
claude --model "$MODEL"
