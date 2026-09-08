#!/bin/bash
# Start llama-server on macOS (Apple Silicon)
# Usage: ./start-server-mac.sh [9b|35b-a3b|qwen3.6-35b-a3b|27b|gemma4-26b|gemma4-31b|gemma4-e4b] [--bench]
#
# Defaults to 35B-A3B (MoE) on 32GB+ RAM, 9B otherwise.
# The 35B-A3B activates only 3B params per token — faster AND smarter than the dense 27B.
# llama.cpp uses Metal automatically on Apple Silicon — no CUDA needed.
#
# Serves BOTH the OpenAI and the Anthropic Messages API (/v1/messages), so Claude
# Code can point straight at it with no LiteLLM proxy. Anthropic tool use needs
# --jinja, which is always passed below.
#
# --bench   pin the exact profile used by the Windows/NVIDIA runs so numbers are
#           comparable across machines: REASONING=0, BATCH=512, UBATCH=256.
#           Without it you get the Mac serving defaults (bigger batches, thinking on).
#
# Tunables (env):
#   REASONING  thinking budget: 0=off (best for the agentic loop), 1536 (or "medium"),
#              -1 (or "max")=unlimited. Default -1 for chat, 0 under --bench.
#   CTX        context window (default 131072; Claude Code's prompt+tools is ~25K alone)
#   BATCH / UBATCH   llama.cpp -b / -ub. Larger = faster prefill, more memory.
#   ALIAS      model name the API advertises (default derived from the size arg).
#              Claude Code's --model must match this.

set -e

# --- flags -------------------------------------------------------------------
BENCH=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --bench) BENCH=1 ;;
    *)       ARGS+=("$a") ;;
  esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

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

if [ "$MODEL_SIZE" = "qwen3.6-35b-a3b" ] || [ "$MODEL_SIZE" = "35b-a3b-3.6" ]; then
  MODEL_NAME="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
  MODEL_REPO="unsloth/Qwen3.6-35B-A3B-GGUF"
  CONTEXT=131072
  DEFAULT_ALIAS="qwen3.6-local"   # same alias the Windows/NVIDIA runs use
  echo "Using Qwen3.6-35B-A3B MoE (current pick; ~22GB at Q4_K_M)"
elif [ "$MODEL_SIZE" = "35b-a3b" ]; then
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

# --- tunables ----------------------------------------------------------------
# --bench pins the profile the Windows/NVIDIA runs use, so cross-machine numbers
# mean something. Otherwise keep the Mac serving defaults.
if [ "$BENCH" = "1" ]; then
  REASONING="${REASONING:-0}"; BATCH="${BATCH:-512}"; UBATCH="${UBATCH:-256}"
  echo "--bench: reasoning=0, -b $BATCH, -ub $UBATCH (matching the NVIDIA runs)"
else
  REASONING="${REASONING:--1}"; BATCH="${BATCH:-4096}"; UBATCH="${UBATCH:-512}"
fi
case "$REASONING" in ""|off|OFF) REASONING=0 ;; medium|MEDIUM) REASONING=1536 ;; max|MAX|full) REASONING=-1 ;; esac
CTX="${CTX:-$CONTEXT}"
ALIAS="${ALIAS:-${DEFAULT_ALIAS:-$MODEL_SIZE}}"

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
echo "Context: $CTX   alias: $ALIAS   reasoning-budget: $REASONING   -b $BATCH -ub $UBATCH"
echo "Log: $LOG"

$LLAMA_SERVER \
  -m "$MODEL_PATH" \
  --host 0.0.0.0 --port $PORT \
  -ngl 99 \
  -c $CTX \
  -b $BATCH \
  -ub $UBATCH \
  -fa on \
  -np 1 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --jinja \
  -a "$ALIAS" \
  --reasoning-budget $REASONING \
  --metrics \
  2>&1 | tee "$LOG"
