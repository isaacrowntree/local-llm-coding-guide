#!/usr/bin/env bash
# benchmark.sh — measure real prefill/decode throughput + peak VRAM for the
# GGUF models you have locally, on YOUR GPU, using the same flags the guide
# recommends for llama-server. Emits a dated markdown table.
#
# The point: this repo's numbers should be *yours*, regenerated on demand,
# never a stale figure copied from a blog. Run it, paste the table.
#
#   ./benchmark.sh                       # bench every *.gguf in $MODELS_DIR
#   ./benchmark.sh ~/models/foo.gguf ... # bench specific files
#
# Config via env:
#   MODELS_DIR (default ~/models)   LLAMA_BENCH (auto-detected)
#   NP=2048  (prompt/prefill tokens)   NG=128 (generated/decode tokens)
#   DEPTH=0  (KV depth to decode at; try 8192 for a realistic-context number)
#   REPS=3   NGL=99   KV=q4_0   FA=1
#   OUT (default benchmarks/RESULTS-$(hostname).md)
set -uo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
NP="${NP:-2048}"; NG="${NG:-128}"; DEPTH="${DEPTH:-0}"
REPS="${REPS:-7}"; NGL="${NGL:-99}"; KV="${KV:-q4_0}"; FA="${FA:-1}"

# --- locate llama-bench -----------------------------------------------------
LLAMA_BENCH="${LLAMA_BENCH:-}"
if [[ -z "$LLAMA_BENCH" ]]; then
  for c in "$(command -v llama-bench 2>/dev/null)" \
           "$HOME/llama.cpp/build/bin/llama-bench" \
           "./llama.cpp/build/bin/llama-bench"; do
    [[ -n "$c" && -x "$c" ]] && { LLAMA_BENCH="$c"; break; }
  done
fi
[[ -x "$LLAMA_BENCH" ]] || { echo "ERROR: llama-bench not found. Set LLAMA_BENCH=/path/to/llama-bench" >&2; exit 1; }

# --- collect models ---------------------------------------------------------
if [[ $# -gt 0 ]]; then MODELS=("$@"); else
  shopt -s nullglob; MODELS=("$MODELS_DIR"/*.gguf); shopt -u nullglob
fi
[[ ${#MODELS[@]} -gt 0 ]] || { echo "ERROR: no .gguf models found in $MODELS_DIR" >&2; exit 1; }

OUT="${OUT:-benchmarks/RESULTS-$(hostname).md}"
mkdir -p "$(dirname "$OUT")"

# --- environment header -----------------------------------------------------
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown GPU")
VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "?")
IDLE_VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
LCPP_REV=$(git -C "$(dirname "$LLAMA_BENCH")/../.." rev-parse --short HEAD 2>/dev/null || echo "?")
DATE=$(date +%Y-%m-%d)

# --- peak-VRAM sampler (background) -----------------------------------------
sample_vram() { # $1 = output file; samples memory.used until killed
  while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 0.3; done > "$1"
}

# --- run one model, echo one markdown row -----------------------------------
bench_one() { # $1 = model path
  local m="$1" name size vlog pp tg peak status
  name=$(basename "$m")
  size=$(du -h "$m" 2>/dev/null | cut -f1)
  vlog=$(mktemp)
  sample_vram "$vlog" & local sampler=$!

  # attempt full offload; on failure retry smaller batch; then partial offload
  local json="" rc=1
  for attempt in "full" "smallbatch" "fit"; do
    local extra=()
    case "$attempt" in
      smallbatch) extra=(-b 1024 -ub 256) ;;
      fit)        extra=(-fitt 1024) ;;   # let llama-bench offload only what fits
    esac
    json=$("$LLAMA_BENCH" -m "$m" -ngl "$NGL" -fa "$FA" -ctk "$KV" -ctv "$KV" \
             -p "$NP" -n "$NG" -d "$DEPTH" -r "$REPS" -o json "${extra[@]}" 2>/dev/null)
    rc=$?
    [[ $rc -eq 0 && -n "$json" ]] && { status="$attempt"; break; }
  done

  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
  peak=$(sort -n "$vlog" 2>/dev/null | tail -1); rm -f "$vlog"
  [[ -z "$peak" ]] && peak="?"

  if [[ $rc -ne 0 || -z "$json" ]]; then
    printf '| %s | %s | — | — | OOM/failed | did not run at ngl=%s |\n' "$name" "$size" "$NGL"
    return
  fi

  # parse: pp = prefill row (n_gen==0), tg = decode row (n_prompt==0).
  # This box shows high run-to-run GPU contention (WSL2 + host desktop), so we
  # report the MEDIAN of the per-rep samples — robust to both stall-lows and
  # glitch-highs that wreck a mean — plus the min–max range for honesty.
  read -r pp tg tgrange < <(printf '%s' "$json" | python3 -c '
import sys,json,statistics
d=json.load(sys.stdin)
def med(r):
    s=r.get("samples_ts") or [r.get("avg_ts")]
    s=[x for x in s if x]; return statistics.median(s) if s else None
def rng(r):
    s=r.get("samples_ts") or []
    s=[x for x in s if x]
    return (min(s),max(s)) if s else None
pp=tg=tgr=None
for r in d:
    npt=r.get("n_prompt",0); ng=r.get("n_gen",0)
    if ng==0 and npt>0: pp=med(r)
    elif ng>0 and npt==0: tg=med(r); tgr=rng(r)
def f(x): return "—" if x is None else f"{x:.0f}"
rs = "" if not tgr else f"{tgr[0]:.0f}-{tgr[1]:.0f}"
print(f(pp), f(tg), rs)
' 2>/dev/null)
  [[ -z "${pp:-}" ]] && { pp="—"; tg="—"; tgrange=""; }

  local note=""; [[ "$status" == "fit" ]] && note="partial offload (does not fully fit)"
  [[ "$status" == "smallbatch" ]] && note="reduced batch"
  printf '| %s | %s | %s | %s | %s | %s | %s |\n' "$name" "$size" "$pp" "$tg" "${tgrange:-—}" "$peak" "$note"
}

# --- emit report ------------------------------------------------------------
{
  echo "## Local benchmark — $GPU (${VRAM_TOTAL} MiB)"
  echo
  echo "- **Date:** $DATE"
  echo "- **GPU / driver:** $GPU, driver $DRIVER, idle VRAM ${IDLE_VRAM} MiB"
  echo "- **llama.cpp:** rev \`$LCPP_REV\` via \`$(basename "$LLAMA_BENCH")\`"
  echo "- **Flags:** \`-ngl $NGL -fa $FA -ctk $KV -ctv $KV\`  •  prefill \`-p $NP\`, decode \`-n $NG\` at depth \`$DEPTH\`, \`-r $REPS\`"
  echo "- **Peak VRAM** = max total GPU memory in use during the run (includes ~${IDLE_VRAM} MiB baseline)."
  echo "- **Median, not mean.** This box shows high run-to-run GPU contention under WSL2 (the Windows host desktop intermittently grabs the GPU). Numbers are the **median** per-rep sample; the range column shows the spread. Decode is at depth \`$DEPTH\` — set \`DEPTH=16384\` for a realistic long-context number (much lower)."
  echo
  echo "| Model | Disk | Prefill tok/s | Decode tok/s (median) | Decode range | Peak VRAM (MiB) | Notes |"
  echo "|-------|------|--------------:|----------------------:|:------------:|----------------:|-------|"
  for m in "${MODELS[@]}"; do
    echo ">> benchmarking $(basename "$m") ..." >&2
    bench_one "$m"
  done
  echo
  echo "_Regenerate with \`./benchmark.sh\`. Numbers are for this exact GPU + llama.cpp build._"
} | tee "$OUT"

echo >&2 "Wrote $OUT"
