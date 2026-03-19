#!/bin/bash
# Start llama-server with Qwen3.5-9B or Nemotron 3 Nano 4B
# Usage: ./start-server.sh [9b|nemotron]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
PORT=8080
LOG="/tmp/llama-server.log"

# Model selection
MODEL_CHOICE="${1:-9b}"
case "$MODEL_CHOICE" in
  nemotron)
    MODEL_PATH="$HOME/models/NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf"
    MODEL_REPO="unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF"
    MODEL_FILE="NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf"
    MODEL_SIZE="2.5GB"
    MODEL_ALIAS="nemotron-3-nano-4b"
    CONTEXT=262144
    ;;
  9b|*)
    MODEL_PATH="$HOME/models/Qwen3.5-9B-Q4_K_M.gguf"
    MODEL_REPO="unsloth/Qwen3.5-9B-GGUF"
    MODEL_FILE="Qwen3.5-9B-Q4_K_M.gguf"
    MODEL_SIZE="5.3GB"
    MODEL_ALIAS="qwen-3.5-9b"
    CONTEXT=131072
    ;;
esac

# Check model exists
if [ ! -f "$MODEL_PATH" ]; then
  echo "Model not found at $MODEL_PATH"
  echo "Downloading $MODEL_FILE ($MODEL_SIZE)..."
  mkdir -p "$HOME/models"
  python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('$MODEL_REPO', '$MODEL_FILE', local_dir='$HOME/models')
"
fi

# Check llama-server exists
if [ ! -f "$LLAMA_SERVER" ]; then
  echo "llama-server not found at $LLAMA_SERVER"
  echo "Building llama.cpp with CUDA..."
  cd "$HOME"
  [ ! -d llama.cpp ] && git clone https://github.com/ggml-org/llama.cpp.git
  cd llama.cpp
  cmake -B build -DGGML_CUDA=ON
  cmake --build build -j$(nproc)
  cd "$SCRIPT_DIR"
fi

# Kill any existing server
pkill -f llama-server 2>/dev/null && sleep 2 || true

echo "Starting llama-server on port $PORT..."
echo "Log: $LOG"

echo "Starting $MODEL_ALIAS on port $PORT (context: $CONTEXT)..."

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
  -a "$MODEL_ALIAS" \
  2>&1 | tee "$LOG"
