#!/bin/bash
# Start vllm-mlx with Gemma 4 on macOS (Apple Silicon)
# Usage: ./start-vllm-mlx-mac.sh [26b-a4b|31b|e4b]
#
# vllm-mlx exposes a native Anthropic /v1/messages API — no proxy needed for Claude Code.
# Uses MLX format models from mlx-community on HuggingFace.
# Defaults to Gemma 4 26B-A4B (MoE, 4B active params).

set -e

PORT=8000
LOG="/tmp/vllm-mlx.log"

# Model selection
MODEL_SIZE="${1:-26b-a4b}"
case "$MODEL_SIZE" in
  31b)
    MODEL="mlx-community/gemma-4-31B-it-4bit"
    echo "Using Gemma 4 31B (dense, ~18GB — tight on 36GB, may swap)"
    ;;
  e4b)
    MODEL="mlx-community/gemma-4-E4B-it-4bit"
    echo "Using Gemma 4 E4B (dense, ~5GB — fast but lower quality)"
    ;;
  26b-a4b|*)
    MODEL="mlx-community/gemma-4-26B-A4B-it-4bit"
    echo "Using Gemma 4 26B-A4B MoE (recommended — 4B active, ~17GB)"
    ;;
esac

# Check vllm-mlx is installed
if ! python3 -c "import vllm_mlx" 2>/dev/null; then
  echo "vllm-mlx not found. Installing..."
  pip install git+https://github.com/AnyLLM/vllm-mlx.git
fi

# Kill any existing vllm-mlx
pkill -f "vllm-mlx" 2>/dev/null && sleep 2 || true

echo ""
echo "Starting vllm-mlx on port $PORT..."
echo "Model: $MODEL (will download on first run)"
echo "Log: $LOG"
echo ""
echo "Endpoints:"
echo "  OpenAI:    http://localhost:$PORT/v1/chat/completions"
echo "  Anthropic: http://localhost:$PORT/v1/messages"
echo ""
echo "To use with Claude Code:"
echo "  ANTHROPIC_BASE_URL=http://localhost:$PORT ANTHROPIC_API_KEY=local claude"
echo ""

vllm-mlx serve "$MODEL" --port $PORT 2>&1 | tee "$LOG"
