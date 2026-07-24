#!/bin/bash
# Start Ollama with Gemma 4 on macOS (Apple Silicon)
# Usage: ./start-ollama-mac.sh [26b-a4b|31b|e4b]
#
# Ollama 0.20+ uses MLX on Apple Silicon — ~31 tok/s generation, ~99 tok/s prefill on M3 Pro.
# Defaults to Gemma 4 26B-A4B (MoE, 4B active params, ~17GB Q4_K_M).
#
# Note: The old llama.cpp Gemma 4 "thinking token" bug (#21321) was a CUDA/ROCm fusion bug,
# not a Metal issue, and was fixed 2026-04-07 (PR #21566). Gemma 4 works in llama.cpp on Mac
# too now — Ollama remains the easiest path, not a required workaround.

set -e

PORT=11434
LOG="/tmp/ollama.log"

# Model selection
MODEL_SIZE="${1:-26b-a4b}"
case "$MODEL_SIZE" in
  31b)
    MODEL="gemma4:31b-it-q4_K_M"
    echo "Using Gemma 4 31B (dense, ~20GB — tight on 36GB, may swap)"
    ;;
  e4b)
    MODEL="gemma4:e4b-it-q4_K_M"
    echo "Using Gemma 4 E4B (dense, ~9.6GB — fast but lower quality)"
    ;;
  26b-a4b|*)
    MODEL="gemma4:26b-a4b-it-q4_K_M"
    echo "Using Gemma 4 26B-A4B MoE (recommended — 4B active, ~17GB, ~31 tok/s)"
    ;;
esac

# Check ollama is installed
if ! command -v ollama &> /dev/null; then
  echo "Ollama not found. Installing via Homebrew..."
  brew install ollama
fi

# Check version
OLLAMA_VERSION=$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "Ollama version: $OLLAMA_VERSION"

# Start ollama serve if not already running
if ! curl -s "http://localhost:$PORT/api/tags" > /dev/null 2>&1; then
  echo "Starting Ollama server with flash attention and q4_0 KV cache..."
  OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q4_0 ollama serve > "$LOG" 2>&1 &
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
echo "  ANTHROPIC_BASE_URL=http://localhost:$PORT ANTHROPIC_AUTH_TOKEN=local claude --model $MODEL"
echo "  (Ollama 0.14+ serves the native Anthropic API — no LiteLLM proxy, no openai/ prefix)"
echo ""

# Run interactively (ctrl-c to stop, server stays running)
ollama run "$MODEL"
