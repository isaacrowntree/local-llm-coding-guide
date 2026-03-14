#!/bin/bash
# Start llama-server with Qwen3.5-9B
# Usage: ./start-server.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="$HOME/models/Qwen3.5-9B-Q4_K_M.gguf"
LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
PORT=8080
LOG="/tmp/llama-server.log"

# Check model exists
if [ ! -f "$MODEL_PATH" ]; then
  echo "Model not found at $MODEL_PATH"
  echo "Downloading Qwen3.5-9B-Q4_K_M.gguf (5.3GB)..."
  mkdir -p "$HOME/models"
  python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-9B-GGUF', 'Qwen3.5-9B-Q4_K_M.gguf', local_dir='$HOME/models')
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

$LLAMA_SERVER \
  -m "$MODEL_PATH" \
  --host 0.0.0.0 --port $PORT \
  -ngl 99 \
  -c 131072 \
  -b 4096 \
  -ub 512 \
  -fa on \
  -np 1 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --reasoning-budget 0 \
  --metrics \
  -a "qwen-3.5-9b" \
  2>&1 | tee "$LOG"
