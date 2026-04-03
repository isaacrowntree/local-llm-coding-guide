#!/bin/bash
# Start TabbyAPI with ExLlamaV3 on Linux/WSL (NVIDIA GPU)
# Usage: ./start-tabby.sh [4bpw|5bpw|6bpw]
#
# ExLlamaV3 uses custom CUDA kernels — ~50-60% faster than llama.cpp on NVIDIA GPUs.
# Uses EXL3 format (CUDA-only). Defaults to Qwen3.5-9B at 4.0 bpw.
#
# Note: Gemma 4 is not yet supported by ExLlamaV3. Use llama.cpp for Gemma 4.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TABBY_DIR="$HOME/tabbyAPI"
PORT=5000
LOG="/tmp/tabbyapi.log"

# Quality selection
QUALITY="${1:-4bpw}"
case "$QUALITY" in
  5bpw)
    MODEL_REPO="turboderp/Qwen3.5-9B-exl3"
    MODEL_REVISION="5.00bpw"
    MODEL_NAME="Qwen3.5-9B-exl3"
    echo "Using Qwen3.5-9B EXL3 at 5.0 bpw (~5.6GB — higher quality)"
    ;;
  6bpw)
    MODEL_REPO="turboderp/Qwen3.5-9B-exl3"
    MODEL_REVISION="6.00bpw"
    MODEL_NAME="Qwen3.5-9B-exl3"
    echo "Using Qwen3.5-9B EXL3 at 6.0 bpw (~6.7GB — highest quality)"
    ;;
  4bpw|*)
    MODEL_REPO="turboderp/Qwen3.5-9B-exl3"
    MODEL_REVISION="4.00bpw"
    MODEL_NAME="Qwen3.5-9B-exl3"
    echo "Using Qwen3.5-9B EXL3 at 4.0 bpw (~4.5GB — recommended)"
    ;;
esac

# Install TabbyAPI if needed
if [ ! -d "$TABBY_DIR" ]; then
  echo "TabbyAPI not found. Cloning..."
  git clone https://github.com/theroyallab/tabbyAPI "$TABBY_DIR"
fi

cd "$TABBY_DIR"

# Download model if needed
if [ ! -d "$TABBY_DIR/models/$MODEL_NAME" ]; then
  echo "Downloading $MODEL_REPO ($MODEL_REVISION)..."
  ./start.sh download "$MODEL_REPO" --revision "$MODEL_REVISION"
fi

# Write config
cat > "$TABBY_DIR/config.yml" << EOF
host: 0.0.0.0
port: $PORT
disable_auth: true

backend: exllamav3
model_dir: models
model_name: $MODEL_NAME

max_seq_len: 32768
cache_size: 32768
cache_mode: Q4

gpu_split_auto: true
reasoning: true
EOF

# Kill any existing TabbyAPI
pkill -f "tabbyAPI" 2>/dev/null && sleep 2 || true
pkill -f "main.py" 2>/dev/null && sleep 2 || true

echo ""
echo "Starting TabbyAPI on port $PORT..."
echo "Model: $MODEL_NAME ($MODEL_REVISION)"
echo "Log: $LOG"
echo ""
echo "To use with Claude Code:"
echo "  ANTHROPIC_BASE_URL=http://localhost:$PORT/v1 ANTHROPIC_AUTH_TOKEN=local claude --model openai/qwen"
echo ""

./start.sh 2>&1 | tee "$LOG"
