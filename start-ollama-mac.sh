#!/bin/bash
# Start Ollama with Gemma 4 on macOS (Apple Silicon)
# Usage: ./start-ollama-mac.sh [26b-a4b|31b|e4b]
#
# Ollama 0.19+ uses MLX on Apple Silicon — ~93% faster decode than the old llama.cpp backend.
# Defaults to Gemma 4 26B-A4B (MoE, 4B active params, 16.9GB Q4).

set -e

PORT=11434
LOG="/tmp/ollama.log"

# Model selection
MODEL_SIZE="${1:-26b-a4b}"
case "$MODEL_SIZE" in
  31b)
    MODEL="gemma4:31b"
    echo "Using Gemma 4 31B (dense, ~18.3GB — tight on 36GB, may swap)"
    ;;
  e4b)
    MODEL="gemma4:e4b"
    echo "Using Gemma 4 E4B (dense, ~5GB — fast but lower quality)"
    ;;
  26b-a4b|*)
    MODEL="gemma4:26b-a4b"
    echo "Using Gemma 4 26B-A4B MoE (recommended — 4B active, ~16.9GB)"
    ;;
esac

# Check ollama is installed
if ! command -v ollama &> /dev/null; then
  echo "Ollama not found. Installing via Homebrew..."
  brew install ollama
fi

# Check version (need 0.19+ for MLX backend)
OLLAMA_VERSION=$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "Ollama version: $OLLAMA_VERSION"

# Start ollama serve if not already running
if ! curl -s "http://localhost:$PORT/api/tags" > /dev/null 2>&1; then
  echo "Starting Ollama server..."
  ollama serve > "$LOG" 2>&1 &
  sleep 3
fi

# Pull model if needed (this is the big download)
echo "Pulling $MODEL (skip if already downloaded)..."
ollama pull "$MODEL"

echo ""
echo "Starting $MODEL..."
echo "API: http://localhost:$PORT/v1 (OpenAI-compatible)"
echo "Log: $LOG"
echo ""
echo "To use with Claude Code:"
echo "  ANTHROPIC_BASE_URL=http://localhost:$PORT/v1 ANTHROPIC_AUTH_TOKEN=local claude --model openai/$MODEL"
echo ""

# Run interactively (ctrl-c to stop, server stays running)
ollama run "$MODEL"
