#!/bin/bash
# Start llama-server on macOS (Apple Silicon)
# Usage: ./start-server-mac.sh [9b|35b-a3b|27b|gemma4-26b|gemma4-31b|gemma4-e4b]
#
# Defaults to 35B-A3B (MoE) on 32GB+ RAM, 9B otherwise.
# The 35B-A3B activates only 3B params per token — faster AND smarter than the dense 27B.
# llama.cpp uses Metal automatically on Apple Silicon — no CUDA needed.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$HOME/models"
PORT=8080
LOG="/tmp/llama-server.log"

# Detect available RAM
TOTAL_RAM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1073741824)}')
echo "Detected ${TOTAL_RAM_GB}GB unified memory"

# Pick model size
MODEL_SIZE="${1:-auto}"
if [ "$MODEL_SIZE" = "auto" ]; then
  if [ "$TOTAL_RAM_GB" -ge 32 ]; then
    MODEL_SIZE="35b-a3b"
  else
    MODEL_SIZE="9b"
  fi
fi

if [ "$MODEL_SIZE" = "35b-a3b" ]; then
  MODEL_NAME="Qwen3.5-35B-A3B-Q4_K_M.gguf"
  MODEL_REPO="unsloth/Qwen3.5-35B-A3B-GGUF"
  CONTEXT=131072
  echo "Using Qwen3.5-35B-A3B MoE (recommended for 32GB+ RAM)"
elif [ "$MODEL_SIZE" = "27b" ]; then
  MODEL_NAME="Qwen3.5-27B-Q4_K_M.gguf"
  MODEL_REPO="unsloth/Qwen3.5-27B-GGUF"
  CONTEXT=131072
  echo "Using Qwen3.5-27B (dense — consider 35B-A3B instead for better speed)"
elif [ "$MODEL_SIZE" = "gemma4-26b" ]; then
  MODEL_NAME="gemma-4-26B-A4B-it-Q4_K_M.gguf"
  MODEL_REPO="unsloth/gemma-4-26B-A4B-it-GGUF"
  CONTEXT=131072
  echo "Using Gemma 4 26B-A4B MoE (4B active, ~17GB)"
elif [ "$MODEL_SIZE" = "gemma4-31b" ]; then
  MODEL_NAME="gemma-4-31B-it-Q4_K_M.gguf"
  MODEL_REPO="unsloth/gemma-4-31B-it-GGUF"
  CONTEXT=131072
  echo "Using Gemma 4 31B (dense, ~18GB — may swap on 36GB)"
elif [ "$MODEL_SIZE" = "gemma4-e4b" ]; then
  MODEL_NAME="gemma-4-E4B-it-Q4_K_M.gguf"
  MODEL_REPO="unsloth/gemma-4-E4B-it-GGUF"
  CONTEXT=131072
  echo "Using Gemma 4 E4B (~5GB — fast but lower quality)"
else
  MODEL_NAME="Qwen3.5-9B-Q4_K_M.gguf"
  MODEL_REPO="unsloth/Qwen3.5-9B-GGUF"
  CONTEXT=131072
  echo "Using Qwen3.5-9B"
fi

MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

# Check model exists
if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading $MODEL_NAME..."
  mkdir -p "$MODEL_DIR"
  python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('$MODEL_REPO', '$MODEL_NAME', local_dir='$MODEL_DIR')
"
fi

# Check llama-server
LLAMA_SERVER=""
if [ -f "$HOME/llama.cpp/build/bin/llama-server" ]; then
  LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
elif command -v llama-server &> /dev/null; then
  LLAMA_SERVER="llama-server"
else
  echo "llama-server not found. Building from source..."
  cd "$HOME"
  [ ! -d llama.cpp ] && git clone https://github.com/ggml-org/llama.cpp.git
  cd llama.cpp
  cmake -B build -DGGML_METAL=ON
  cmake --build build -j$(sysctl -n hw.ncpu)
  LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
  cd "$SCRIPT_DIR"
fi

# Kill any existing server
pkill -f llama-server 2>/dev/null && sleep 2 || true

echo "Starting llama-server on port $PORT..."
echo "Model: $MODEL_NAME"
echo "Context: $CONTEXT"
echo "Log: $LOG"

$LLAMA_SERVER \
  -m "$MODEL_PATH" \
  --host 0.0.0.0 --port $PORT \
  -ngl 99 \
  -c $CONTEXT \
  -b 4096 \
  -ub 512 \
  -fa on \
  -np 1 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --reasoning-budget -1 \
  --metrics \
  2>&1 | tee "$LOG"
