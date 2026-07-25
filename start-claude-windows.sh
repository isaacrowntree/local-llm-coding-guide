#!/bin/bash
# Launch Claude Code against a local model on Windows/WSL2 + NVIDIA (llama.cpp).
#
# Why this exists: on a 12GB card the best local coding model (Qwen3.6-35B-A3B
# at IQ2) uses ~all of VRAM at full GPU offload (free=0) and is UNSTABLE for
# agentic use — it loads, then OOMs the moment the agent's context or a CUDA
# graph needs more. The fix is --n-cpu-moe (-ncmoe): keep some MoE expert layers
# on the CPU to buy VRAM headroom. Measured: -ncmoe 14 -> ~8.6GB used, stable,
# ~44 tok/s decode. This script wires up llama-server (with headroom) + LiteLLM
# (Anthropic bridge) + Claude Code.
#
# Usage:
#   ./start-claude-windows.sh                              # interactive
#   ./start-claude-windows.sh --dangerously-skip-permissions
#   ./start-claude-windows.sh -p "fix the failing test"    # headless
#   (any args after the script name are passed straight to `claude`)
#
# Tunables (env):
#   MODEL_PATH  default ~/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
#   NCMOE       expert layers to keep on CPU (default 14; raise for more VRAM headroom)
#   CTX         context window (default 24576; raise CTX -> also raise NCMOE)
#   REASONING   thinking budget: 0=off (default, best for the tool loop),
#               1536 (or "medium"), -1 (or "max")=unlimited. Thinking is
#               context-hungry and slow per turn — off is best for agentic use.
#   LITELLM_BIN override the litellm binary (defaults to ~/litellm-venv/bin/litellm)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf}"
MODEL_NAME="qwen3.6-local"        # what you pass to `claude --model`
NCMOE="${NCMOE:-16}"
CTX="${CTX:-40960}"   # Claude Code's system prompt + tools alone is ~25K tokens
# --reasoning-budget: 0 = off (fastest, best for the agentic loop), 1536 ~= "medium",
# -1 = unlimited (smart but slow; can flood context). Default off.
REASONING="${REASONING:-0}"
PORT=8080
LITELLM_PORT=4000
CONFIG="$SCRIPT_DIR/litellm-claude.yaml"
SRV_LOG="/tmp/llama-server-claude.log"
LITELLM_LOG="/tmp/litellm-claude.log"

case "$REASONING" in ""|off|OFF) REASONING=0 ;; medium|MEDIUM) REASONING=1536 ;; max|MAX|full) REASONING=-1 ;; esac

command -v claude >/dev/null 2>&1 || { echo "ERROR: 'claude' (Claude Code) not on PATH."; exit 1; }
# Prefer the venv litellm — Ubuntu 26.04's Python 3.14 removed older system installs.
LITELLM_BIN="${LITELLM_BIN:-}"
if [ -z "$LITELLM_BIN" ]; then
  if [ -x "$HOME/litellm-venv/bin/litellm" ]; then LITELLM_BIN="$HOME/litellm-venv/bin/litellm"
  elif command -v litellm >/dev/null 2>&1; then LITELLM_BIN="litellm"
  else echo "ERROR: litellm not found. Install: python3 -m venv ~/litellm-venv && ~/litellm-venv/bin/pip install 'litellm[proxy]'"; exit 1; fi
fi
[ -f "$LLAMA_SERVER" ] || { echo "ERROR: llama-server not found at $LLAMA_SERVER"; exit 1; }
[ -f "$MODEL_PATH" ]   || { echo "ERROR: model not found at $MODEL_PATH"; exit 1; }

STARTED_SERVER=0
STARTED_LITELLM=0
cleanup() {
  echo ""
  [ "$STARTED_LITELLM" = "1" ] && { echo "Stopping LiteLLM..."; kill "$LITELLM_PID" 2>/dev/null || true; }
  [ "$STARTED_SERVER" = "1" ]  && { echo "Stopping llama-server..."; kill "$SRV_PID" 2>/dev/null || true; }
  exit 0
}
trap cleanup INT TERM

# --- 1. llama-server (start only if not already serving) --------------------
if curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
  echo "Reusing llama-server already on :$PORT"
else
  # ensure a clean GPU — a previous instance at free=0 can leave residue that OOMs the new load
  pkill -f "build/bin/llama-server" 2>/dev/null && sleep 5 || true
  echo "Starting llama-server: $(basename "$MODEL_PATH")"
  echo "  -ncmoe $NCMOE  -c $CTX  reasoning-budget=$REASONING (0=off, 1536~medium, -1=max)"
  nohup "$LLAMA_SERVER" -m "$MODEL_PATH" \
    --host 127.0.0.1 --port $PORT \
    -ngl 99 -ncmoe "$NCMOE" -c "$CTX" -b 512 -ub 256 -fa on -np 1 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --reasoning-budget "$REASONING" -a "$MODEL_NAME" \
    > "$SRV_LOG" 2>&1 &
  SRV_PID=$!; STARTED_SERVER=1
  echo -n "  waiting for model load"
  for i in $(seq 1 40); do
    grep -q "server is listening" "$SRV_LOG" 2>/dev/null && { echo " ready."; break; }
    kill -0 "$SRV_PID" 2>/dev/null || { echo ""; echo "llama-server died — see $SRV_LOG"; tail -5 "$SRV_LOG"; exit 1; }
    echo -n "."; sleep 3
  done
  VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "  VRAM in use: ${VRAM} MiB"
fi

# --- 2. LiteLLM proxy (Anthropic <-> OpenAI bridge) -------------------------
if curl -s "http://localhost:$LITELLM_PORT/health/readiness" >/dev/null 2>&1; then
  echo "Reusing LiteLLM already on :$LITELLM_PORT"
else
  echo "Starting LiteLLM proxy on :$LITELLM_PORT..."
  nohup "$LITELLM_BIN" --config "$CONFIG" --port $LITELLM_PORT --host 127.0.0.1 > "$LITELLM_LOG" 2>&1 &
  LITELLM_PID=$!; STARTED_LITELLM=1
  for i in $(seq 1 20); do
    curl -s "http://localhost:$LITELLM_PORT/health/readiness" >/dev/null 2>&1 && { echo "  ready."; break; }
    kill -0 "$LITELLM_PID" 2>/dev/null || { echo "LiteLLM died — see $LITELLM_LOG"; exit 1; }
    sleep 2
  done
fi

# --- 3. Claude Code ---------------------------------------------------------
echo ""
echo "============================================================"
echo "  Claude Code -> LiteLLM(:$LITELLM_PORT) -> llama-server(:$PORT)"
echo "  Model: $MODEL_NAME   (reasoning-budget $REASONING)"
echo ""
echo "  Copy-paste to run it yourself in any project dir:"
echo ""
echo "    ANTHROPIC_BASE_URL=http://localhost:$LITELLM_PORT \\"
echo "    ANTHROPIC_AUTH_TOKEN=local ANTHROPIC_API_KEY=local \\"
echo "    claude --model $MODEL_NAME --dangerously-skip-permissions \\"
echo "      --disallowedTools WebSearch WebFetch"
echo "============================================================"
echo ""

# WebSearch/WebFetch are Anthropic *server-side* tools (tool type != "function");
# llama-server rejects them with a 400, and they can't run locally anyway. Disable them.
ANTHROPIC_BASE_URL="http://localhost:$LITELLM_PORT" \
ANTHROPIC_AUTH_TOKEN="local" \
ANTHROPIC_API_KEY="local" \
claude --model "$MODEL_NAME" --disallowedTools WebSearch WebFetch "$@"

cleanup
