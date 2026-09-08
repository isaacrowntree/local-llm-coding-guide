#!/bin/bash
# Benchmark Claude Code driving local models (macOS Apple Silicon, Ollama backend)
#
# Usage:
#   ./bench-claude-code.sh                          # all default models, all tasks
#   ./bench-claude-code.sh --models "qwen3.8:27b-q4_K_M"
#   ./bench-claude-code.sh --tasks "01 04" --turns 40
#   ./bench-claude-code.sh --list                   # show candidates + disk cost
#   ./bench-claude-code.sh --pull-only              # just download the models
#
# Each (model, task) pair runs Claude Code headless in a throwaway copy of a
# fixture repo, then an external verify.sh decides pass/fail. Results land in
# bench/results/<timestamp>/results.csv plus each run's full JSON + transcript.
#
# Claude Code speaks the Anthropic Messages API; Ollama 0.14+ serves it natively,
# so there is no LiteLLM proxy here. See the Claude Code section of README.md.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$ROOT/bench/tasks"
PORT=11434
BASE_URL="http://localhost:$PORT"
MANAGE_OLLAMA=1      # --base-url turns this off: the server is yours to start, not ours
OLLAMA_LOG="/tmp/ollama-bench.log"

# Candidates for a 36GB Apple Silicon machine. Each ~17-20GB on disk.
DEFAULT_MODELS=(
  "qwen3.8:27b-mtp-q4_K_M"          # Aug 2026 quality leader + MTP speculative decoding
  "qwen3.8:27b-q4_K_M"              # same weights, MTP off — isolates the MTP speedup
  "muse-glimmer:30b-q4_K_M-dflash"  # Meta agentic model + DFlash drafter
  "gemma4:26b-a4b-it-q4_K_M"        # incumbent baseline from the README table
)

MODELS=()
TASKS=()
TURNS=60
TIMEOUT=1800         # per-run wall clock cap, seconds. Each agent turn re-prefills the
                     # ~20k system prompt, so turns cost 100s (MoE) to 250s (dense 27B).
CTX=32768            # Claude Code's system prompt + tools measured at ~20k tokens on its own,
                     # so 4096 (Ollama default) is hopeless. 65536 was worse: the KV cache on top
                     # of 17GB of weights pushed a 36GB machine into swap (6.2GB swap used).
KEEP_ALIVE="60m"     # weights stay resident for a model's whole pass — no mid-pass reloads
KEEP_WORKDIRS=0
RM_AFTER=0           # --rm-after: delete each model from disk after its pass
BAIL_SLOW=0          # --bail-slow N: give up on a model once any single task exceeds N seconds
                     # (or times out). Gemma 4 26B-A4B's slowest task was 195s, so ~600s means
                     # "3x slower than a known-good model" — not worth hours to confirm.
PULL_ONLY=0
LIST_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --models)    IFS=' ' read -r -a MODELS <<< "$2"; shift 2 ;;
    --tasks)     IFS=' ' read -r -a TASKS  <<< "$2"; shift 2 ;;
    --turns)     TURNS="$2"; shift 2 ;;
    --timeout)   TIMEOUT="$2"; shift 2 ;;
    --ctx)       CTX="$2"; shift 2 ;;
    --keep)      KEEP_WORKDIRS=1; shift ;;
    --rm-after)  RM_AFTER=1; shift ;;
    --bail-slow) BAIL_SLOW="$2"; shift 2 ;;
    --base-url)  BASE_URL="$2"; MANAGE_OLLAMA=0; shift 2 ;;
    --pull-only) PULL_ONLY=1; shift ;;
    --list)      LIST_ONLY=1; shift ;;
    -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
    *)           echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

[ ${#MODELS[@]} -eq 0 ] && MODELS=("${DEFAULT_MODELS[@]}")

if [ ${#TASKS[@]} -eq 0 ]; then
  while IFS= read -r d; do TASKS+=("$(basename "$d")"); done < <(find "$TASK_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
else
  # Allow short task ids: "01" -> "01-fix-bug"
  expanded=()
  for t in "${TASKS[@]}"; do
    match=$(find "$TASK_DIR" -mindepth 1 -maxdepth 1 -type d -name "${t}*" | sort | head -1)
    [ -n "$match" ] && expanded+=("$(basename "$match")") || echo "warn: no task matching '$t'" >&2
  done
  TASKS=("${expanded[@]}")
fi

if [ "$LIST_ONLY" = 1 ]; then
  echo "Models:"; printf '  %s\n' "${MODELS[@]}"
  echo "Tasks:";  printf '  %s\n' "${TASKS[@]}"
  echo
  echo "Runs: $(( ${#MODELS[@]} * ${#TASKS[@]} ))  (max ~$(( ${#MODELS[@]} * ${#TASKS[@]} * TIMEOUT / 60 )) min worst case)"
  exit 0
fi

REQUIRED_BINS="claude jq python3"
[ "$MANAGE_OLLAMA" = 1 ] && REQUIRED_BINS="ollama $REQUIRED_BINS"
for bin in $REQUIRED_BINS; do
  command -v "$bin" >/dev/null || { echo "missing required tool: $bin" >&2; exit 1; }
done

# --- non-Ollama backend: just check it answers ------------------------------
if [ "$MANAGE_OLLAMA" = 0 ]; then
  echo "Using external server at $BASE_URL (not managing it)."
  echo "  Ensure it is running, has the model loaded, and speaks the Anthropic Messages API."
  if ! curl -s -o /dev/null -m 5 "$BASE_URL/v1/models" && ! curl -s -o /dev/null -m 5 "$BASE_URL"; then
    echo "WARNING: $BASE_URL did not respond — runs will fail." >&2
  fi
fi

# --- disk check -------------------------------------------------------------
if [ "$MANAGE_OLLAMA" = 1 ]; then
AVAIL_GB=$(/bin/df -g /System/Volumes/Data | awk 'NR==2 {print $4}')
NEED_GB=$(( ${#MODELS[@]} * 21 ))
echo "Disk: ${AVAIL_GB}GB free, models need up to ~${NEED_GB}GB"
if [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
  echo "WARNING: may run out of space. Free some up, or run fewer --models." >&2
fi
fi

# --- ollama server ----------------------------------------------------------
# The context length matters more than anything else here: Claude Code's system
# prompt plus tool definitions blows past Ollama's 4096 default, and the model
# silently truncates instead of failing loudly.
if [ "$MANAGE_OLLAMA" = 0 ]; then
  :
elif curl -s "$BASE_URL/api/tags" >/dev/null 2>&1; then
  echo "Ollama already running on :$PORT — assuming OLLAMA_CONTEXT_LENGTH >= $CTX."
  echo "  (If runs fail oddly, kill it and re-run so this script can set the context.)"
else
  echo "Starting Ollama (ctx=$CTX, flash attention, q4_0 KV cache)..."
  OLLAMA_FLASH_ATTENTION=1 \
  OLLAMA_KV_CACHE_TYPE=q4_0 \
  OLLAMA_CONTEXT_LENGTH="$CTX" \
  OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_KEEP_ALIVE="$KEEP_ALIVE" \
  ollama serve > "$OLLAMA_LOG" 2>&1 &
  for _ in $(seq 1 30); do
    curl -s "$BASE_URL/api/tags" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -s "$BASE_URL/api/tags" >/dev/null 2>&1 || { echo "Ollama failed to start; see $OLLAMA_LOG" >&2; exit 1; }
fi

if [ "$PULL_ONLY" = 1 ]; then
  for m in "${MODELS[@]}"; do
    echo "Pulling $m ..."
    ollama pull "$m" || { echo "pull failed: $m" >&2; exit 1; }
  done
  echo "Pull complete."
  exit 0
fi

# --- results ----------------------------------------------------------------
STAMP=$(date +%Y%m%d-%H%M%S)
RESULT_DIR="$ROOT/bench/results/$STAMP"
mkdir -p "$RESULT_DIR"
CSV="$RESULT_DIR/results.csv"
echo "model,task,pass,duration_s,turns,input_tokens,output_tokens,eff_tok_per_s,note" > "$CSV"

slug() { echo "$1" | tr '/:.' '---'; }

echo
echo "=== ${#MODELS[@]} models x ${#TASKS[@]} tasks -> $RESULT_DIR ==="

for model in "${MODELS[@]}"; do
  mslug=$(slug "$model")
  echo
  echo "### $model"

  # Pull just-in-time so a failure on model 3 doesn't cost you models 1-2 of disk.
  if [ "$MANAGE_OLLAMA" = 1 ] && ! ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$model"; then
    echo "  pulling $model ..."
    ollama pull "$model" || { echo "  pull failed, skipping $model" >&2; continue; }
  fi

  # Load the weights ONCE for this model's entire pass. (Ollama-specific warmup;
  # an external server is expected to have the model resident already.) keep_alive is long enough
  # that verification/setup gaps between tasks never trigger an unload — a reload
  # would cost ~30-60s per task on an 18GB model.
  load_start=$(python3 -c 'import time; print(time.time())')
  [ "$MANAGE_OLLAMA" = 1 ] && curl -s "$BASE_URL/api/generate" \
    -d "{\"model\":\"$model\",\"prompt\":\"hi\",\"stream\":false,\"keep_alive\":\"$KEEP_ALIVE\"}" >/dev/null 2>&1
  load_end=$(python3 -c 'import time; print(time.time())')
  load_s=$(python3 -c "print(f'{$load_end - $load_start:.1f}')")
  echo "  model load: ${load_s}s"

  for task in "${TASKS[@]}"; do
    src="$TASK_DIR/$task"
    [ -f "$src/PROMPT.md" ] || { echo "  skip $task (no PROMPT.md)"; continue; }

    work="$RESULT_DIR/work/$mslug/$task"
    rm -rf "$work"; mkdir -p "$work"
    bash "$src/setup.sh" "$work" || { echo "  setup failed: $task"; continue; }

    json_out="$RESULT_DIR/$mslug--$task.json"
    printf '  %-20s ' "$task"

    start=$(python3 -c 'import time; print(time.time())')

    # macOS has no coreutils `timeout`, and a sleep-based watchdog proved
    # unreliable for long background runs — perl's alarm() is dependable.
    (
      cd "$work" || exit 1
      exec perl -e '
        my $timeout = shift;
        my $pid = fork();
        die "fork failed" unless defined $pid;
        if ($pid == 0) { exec @ARGV; exit 127; }
        $SIG{ALRM} = sub {
          kill "TERM", $pid; sleep 5; kill "KILL", $pid; exit 124;
        };
        alarm $timeout;
        waitpid($pid, 0);
        exit($? >> 8);
      ' "$TIMEOUT" \
      env -u ANTHROPIC_API_KEY \
        ANTHROPIC_BASE_URL="$BASE_URL" \
        ANTHROPIC_AUTH_TOKEN="local" \
        ANTHROPIC_MODEL="$model" \
        ANTHROPIC_SMALL_FAST_MODEL="$model" \
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
        CLAUDE_CODE_ATTRIBUTION_HEADER=0 \
        claude -p "$(cat "$src/PROMPT.md")" \
          --model "$model" \
          --output-format json \
          --max-turns "$TURNS" \
          --dangerously-skip-permissions
    ) > "$json_out" 2>"$RESULT_DIR/$mslug--$task.stderr"
    rc=$?

    end=$(python3 -c 'import time; print(time.time())')
    dur=$(python3 -c "print(f'{$end - $start:.1f}')")

    note=""
    if [ $rc -eq 124 ]; then
      note="TIMEOUT after ${TIMEOUT}s"
    elif [ $rc -ge 128 ]; then
      note="killed by signal $((rc - 128))"
    elif [ $rc -ne 0 ]; then
      note="claude exited $rc"
    fi

    turns=$(jq -r '.num_turns // ""'              "$json_out" 2>/dev/null)
    in_tok=$(jq -r '.usage.input_tokens // ""'    "$json_out" 2>/dev/null)
    out_tok=$(jq -r '.usage.output_tokens // ""'  "$json_out" 2>/dev/null)
    api_ms=$(jq -r '.duration_api_ms // .duration_ms // ""' "$json_out" 2>/dev/null)
    [ "$(jq -r '.is_error // false' "$json_out" 2>/dev/null)" = "true" ] && note="${note:+$note; }api error"

    tps=""
    if [ -n "$out_tok" ] && [ -n "$api_ms" ] && [ "$api_ms" != "0" ]; then
      tps=$(python3 -c "print(f'{$out_tok / ($api_ms/1000):.1f}')" 2>/dev/null)
    fi

    if bash "$src/verify.sh" "$work" > "$RESULT_DIR/$mslug--$task.verify" 2>&1; then
      pass=PASS; echo "PASS  ${dur}s  ${turns:-?} turns  ${tps:-?} tok/s"
    else
      pass=FAIL
      reason=$(head -1 "$RESULT_DIR/$mslug--$task.verify")
      echo "FAIL  ${dur}s  ${turns:-?} turns  — ${reason:-no reason}"
      note="${note:+$note; }${reason}"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,"%s"\n' \
      "$model" "$task" "$pass" "$dur" "$turns" "$in_tok" "$out_tok" "$tps" \
      "$(echo "$note" | tr -d '"' | cut -c1-120)" >> "$CSV"

    # Early bail: one hopeless task is enough evidence — skip this model's remainder.
    if [ "$BAIL_SLOW" != 0 ]; then
      too_slow=$(python3 -c "print(1 if $dur > $BAIL_SLOW else 0)" 2>/dev/null)
      if [ "$too_slow" = "1" ] || [ $rc -eq 124 ]; then
        echo "  BAILING on $model — '$task' took ${dur}s (limit ${BAIL_SLOW}s). Skipping its remaining tasks."
        for skipped in "${TASKS[@]}"; do
          [ "$skipped" = "$task" ] && seen_current=1 && continue
          [ "${seen_current:-0}" = 1 ] && printf '%s,%s,SKIP,,,,,,"bailed: %s exceeded %ss"\n' \
            "$model" "$skipped" "$task" "$BAIL_SLOW" >> "$CSV"
        done
        unset seen_current
        break
      fi
    fi
  done

  # Unload before the next model so two sets of weights never share 36GB.
  [ "$MANAGE_OLLAMA" = 1 ] && curl -s "$BASE_URL/api/generate" -d "{\"model\":\"$model\",\"keep_alive\":0}" >/dev/null 2>&1
  sleep 2
  if [ "$RM_AFTER" = 1 ] && [ "$MANAGE_OLLAMA" = 1 ]; then
    echo "  removing $model from disk (--rm-after)"
    ollama rm "$model" >/dev/null 2>&1
  fi
done

[ "$KEEP_WORKDIRS" = 0 ] && rm -rf "$RESULT_DIR/work"

echo
echo "=== Summary ==="
python3 - "$CSV" <<'PY'
import csv, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("no results"); sys.exit(0)

tasks = sorted({r["task"] for r in rows})
by_model = defaultdict(dict)
for r in rows:
    by_model[r["model"]][r["task"]] = r

w = max(len(m) for m in by_model) + 2
hdr = "Model".ljust(w) + "".join(t[:14].ljust(16) for t in tasks) + "Pass".rjust(7) + "Eff tok/s".rjust(11)
print(hdr); print("-" * len(hdr))
for model, res in by_model.items():
    line = model.ljust(w)
    npass = 0
    speeds = []
    for t in tasks:
        r = res.get(t)
        if not r:
            line += "-".ljust(16); continue
        ok = r["pass"] == "PASS"
        npass += ok
        line += (("PASS " if ok else "FAIL ") + f'{float(r["duration_s"]):.0f}s').ljust(16)
        try:
            speeds.append(float(r["eff_tok_per_s"]))
        except (ValueError, TypeError):
            pass
    line += f"{npass}/{len(tasks)}".rjust(7)
    line += (f"{sum(speeds)/len(speeds):.1f}" if speeds else "-").rjust(11)
    print(line)
PY

echo
echo "CSV:      $CSV"
echo "Raw JSON: $RESULT_DIR/"
