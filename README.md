# Local LLM Coding Guide

Run local LLMs as a coding assistant on consumer hardware (NVIDIA/CUDA or Apple Silicon) via llama.cpp, Ollama, ExLlamaV3, mlx-lm, or vllm-mlx.

Tested on:
- **Windows/WSL2:** RTX 4070 Ti (12GB), Intel Core Ultra 9 285K, 48GB DDR5
- **macOS:** M3 MacBook Pro, 36GB unified memory

> **Why this guide is structured the way it is.** Specific model names and tok/s numbers go stale fast — a new "best model for 12GB" lands every few weeks. So the recommendations here are **rules, not names**: [How to choose](#how-to-choose-start-here) is stable, the model names that satisfy those rules today live in one dated [Current picks](#current-picks) block, and every performance number is something you regenerate for *your own* GPU with [`./benchmark.sh`](#benchmark-it-yourself) rather than trust from a table. When a better model drops, you swap the pick — the framework doesn't move.

## How to choose (start here)

Three stable questions decide your setup. The *rule* in each step rarely changes; only which model satisfies it does (see [Current picks](#current-picks)).

**1. Platform & memory budget** — this is your hard constraint.

| Platform | Budget = | Tiers |
|----------|----------|-------|
| **NVIDIA (CUDA)** | your VRAM | 8 / 12 / 16 / 24GB+ |
| **Apple Silicon** | unified memory shared with the OS — usable ≈ *(total − 8GB)* | 16 / 24 / 36 / 64GB+ |

**2. Goal** — what you're optimizing for:

| Goal | Rule |
|------|------|
| **Quality** | the *largest-total-parameter MoE that still fits your budget at ≥IQ2*, with room for a usable context. MoE gives you big-model reasoning at small-model speed. |
| **Speed** (tab-completion, quick edits) | a *small-active-param MoE at Q4* — maximize tok/s, accept lower ceiling. |
| **Long context** | the model whose *KV cache at your target window fits alongside the weights* with `q4_0` cache. Decode is memory-bandwidth bound, so smaller weights buy you more context. |

**3. Engine** — follows from platform (details in [Inference Engines](#inference-engines-nvidia-cuda)):

| Platform | Default | Max single-user speed | Serving / batched |
|----------|---------|----------------------|-------------------|
| **NVIDIA** | llama.cpp (universal, day-0 models) | ExLlamaV3 + TabbyAPI (EXL3) | TensorRT-LLM |
| **Apple** | Ollama (MLX, native Anthropic API) | mlx-lm | vllm-mlx (native Anthropic, no proxy) |

**Selection rules by tier** (stable — the pick that fills each cell is in [Current picks](#current-picks)):

| Budget | Quality rule | Speed rule |
|--------|-------------|-----------|
| **8GB VRAM / ~16GB unified** | best ~4B-active MoE at Q4 | fastest ~2–4B-active MoE at Q4 |
| **12GB VRAM / ~24GB unified** | best 30–40B-total A3B MoE at IQ2–IQ3 | small MoE at Q4 |
| **16GB VRAM / ~32GB unified** | same 30–40B A3B MoE at IQ3–Q4 (longer ctx) | small MoE at Q4 |
| **24GB VRAM / 36GB+ unified** | 30B-class at Q4, or A3B MoE at full Q4 | A3B MoE at Q4 |

Then **verify the current pick fits and performs** on your actual hardware with [`./benchmark.sh`](#benchmark-it-yourself) — that is the whole point of this repo. Don't trust the numbers below; regenerate them.

## Performance

> The table below is **our** last local run, not a spec sheet. Regenerate it for your GPU with [`./benchmark.sh`](#benchmark-it-yourself); results land in `benchmarks/RESULTS-<host>.md`. Numbers move with your llama.cpp build, drivers, and flags.

### Windows / NVIDIA — RTX 4070 Ti 12GB (measured 2026-07-25 via `benchmark.sh`)

Median decode tok/s (this box has high run-to-run variance under WSL2 — see [full results](benchmarks/RESULTS-IsaacPC.md)). `d0` = decode at empty context (peak); `@16K` = decode at 16K context, which is what you actually feel while coding. Flags: `-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0`.

| Model (GGUF) | Prefill tok/s | Decode tok/s (d0) | Decode @16K | Peak VRAM |
|--------------|--------------:|------------------:|------------:|----------:|
| **Qwen3.6-35B-A3B UD-IQ2_M** | 3392 | **113** | **~27** | 11.9GB |
| **Gemma 4 E4B Q4_K_M** | 7049 | **93** | ~78 | 4.2GB |
| DeepSeek-R1-Qwen3-8B Q4_K_M | 5246 | 76 | — | 5.6GB |
| Qwen3.5-9B Q4_K_M | 4531 | 67 | — | 6.2GB |
| Qwen3-14B Q4_K_M | 2910 | 53 | — | 9.4GB |
| Devstral-Small-2 24B IQ2_M | 1646 | 46 | — | 8.6GB |

> **This is why we benchmark ourselves.** The old guide listed Qwen3.6-A3B at "~30-38 tok/s" — that was really the *16K-context* number (~27); at low context it decodes **~113**. Both are real, so the table now shows both. Peak VRAM 11.9GB confirms it fits 12GB, but with almost no headroom — expect OOM if you push context past ~57K or run anything else on the GPU. Other NVIDIA cards (3060/3090/etc.): run `./benchmark.sh` — don't trust a number measured on a different card.

### macOS — M3 Pro 36GB (measured 2026-07-24)

| Model | Tok/s | Context | Memory Used |
|-------|-------|---------|-------------|
| **Qwen3.6-35B-A3B Q4_K_M (Ollama MLX)** | **~35** | 262K | **~22GB** |
| **Qwen3.6-35B-A3B 4bit (mlx-lm)** | **~48** † | 262K | **~20GB** |
| Qwen3.5-35B-A3B Q4_K_M | ~29 | 131K | ~22GB |
| Qwen3.5-9B Q4_K_M | ~20 | 131K | ~7GB |
| Qwen3.5-27B Q4_K_M | ~9 * | 131K | ~18GB |
| **Gemma 4 26B-A4B Q4_K_M (Ollama MLX)** | **~33** | 256K | **~17GB** |
| Gemma 4 31B Q4_K_M (dense, Ollama MLX) | **~6** ‡ | 256K | ~18GB |
| Gemma 4 31B + MTP | n/a § | 256K | — |
| **Qwen3.8-27B Q4_K_M (dense, Ollama)** | ~5 ¶ | 256K | ~17GB |
| Muse Glimmer 30B Q4_K_M (dense, Ollama) | n/a ‖ | 128K | ~18GB |

*The dense 27B model is slower than the 35B-A3B on 36GB machines due to higher memory bandwidth requirements. The 35B-A3B (MoE) is faster *and* smarter — see [Why MoE?](#why-moe-mixture-of-experts) below.

† Measured 2026-07-24 on M3 Pro 36GB. Raw `mlx-lm` decodes Qwen 3.6 35B-A3B at **~48 tok/s** vs Ollama MLX's **~35 tok/s** (both matched over a ~4000-token generation) — Ollama trades ~30% decode speed for its `q4_0` KV-cache + serving convenience. Ollama's prefill is much higher (~145 vs an unreliable CLI figure for mlx-lm). Pick Ollama for ease, `mlx-lm` if you want the fastest single-stream decode.

‡ The dense 31B **swap-thrashes on 36GB** (~18GB weights leaves too little headroom) — measured ~6 tok/s, i.e. 5× slower than the 26B-A4B MoE. Not recommended on 36GB; use the 26B-A4B MoE instead.

¶ Measured 2026-08-25 driving **Claude Code** (see [Benchmarking](#benchmarking-claude-code-harness)). Dense 27B loses on both axes vs the 26B-A4B MoE: **~82 tok/s prefill** (vs ~720–830) and **~5 tok/s decode** (vs ~25). A 21k-token cold start costs ~262s once per session, then generation crawls. It timed out at 900s without finishing a task the MoE completed in 89s. The `mtp` tag does not rescue it.

‖ Muse Glimmer failed all attempted tasks, but on **tool calling, not speed** — it returned after 1–2 turns without invoking a single tool. Looks like an Ollama chat-template issue; retest on a newer Ollama before drawing conclusions about the model.

§ Ollama ships Gemma-4 MTP **only as `gemma4:31b-coding-mtp-bf16` (~62GB bf16)** — there is no quantized MTP tag, and 62GB cannot fit 36GB unified memory. The MTP speedup is therefore **not usable on a 36GB Mac** via Ollama. (llama.cpp's Q4 MTP path for Qwen is a separate option — see the MTP section.)

## Benchmark it yourself

The numbers in this guide are generated, not asserted. [`benchmark.sh`](benchmark.sh) runs `llama-bench` over the GGUFs you have locally, using the same flags the guide recommends for `llama-server`, and prints a dated markdown table of prefill/decode throughput and peak VRAM — for **your** GPU.

```bash
# Bench every *.gguf in ~/models (override with MODELS_DIR)
./benchmark.sh

# Or specific files
./benchmark.sh ~/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf ~/models/gemma-4-E4B-it-Q4_K_M.gguf
```

Config via env: `NP` (prefill tokens, default 2048), `NG` (decode tokens, 128), `DEPTH` (KV depth to decode at — set `8192` for a realistic long-context decode number), `REPS` (3), `NGL` (99), `KV` (`q4_0`), `FA` (1). Results are teed to `benchmarks/RESULTS-<host>.md`. Models too big to fully offload are retried with a smaller batch, then partial offload, and flagged in the Notes column.

This is the antidote to staleness: when a new model shows up, download the GGUF, run `./benchmark.sh`, and you have a trustworthy tok/s + VRAM number for *your* card in under a minute — no waiting for someone to re-benchmark it on hardware unlike yours.

## Quick Start

### 1. Install llama.cpp

**Option A: Pre-built binary (recommended)**

Download the latest release for your platform from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases).

**Option B: Build from source (CUDA / Metal)**

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Linux/WSL (NVIDIA GPU):
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89"  # 89=4070Ti, 86=3090/3060
cmake --build build -j$(nproc)

# macOS (Apple Silicon):
cmake -B build -DGGML_METAL=ON
cmake --build build -j$(sysctl -n hw.ncpu)
```

### 2. Download the model

**For macOS with 32GB+ RAM (recommended — Qwen 3.6 is the April 2026 update):**
```bash
pip install huggingface-hub
huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --local-dir ./models
```

Same MoE shape as Qwen 3.5 35B-A3B (35B total / 3B active per token, ~22GB at Q4_K_M) but with significantly better coding scores — 73.4% on SWE-bench Verified. Drop-in replacement.

**Qwen 3.5 35B-A3B** (still solid, slightly smaller download):
```bash
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-Q4_K_M.gguf --local-dir ./models
```

**For NVIDIA GPUs (12GB VRAM) — Qwen3.6-35B-A3B (Sonnet-class, 8K context):**
```bash
huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-IQ2_M.gguf --local-dir ./models
```

**For NVIDIA GPUs (8-12GB VRAM) — Gemma 4 E4B (fast, 131K context):**
```bash
huggingface-cli download unsloth/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q4_K_M.gguf --local-dir ./models
```

**For NVIDIA GPUs (8-12GB VRAM) — Qwen3.5-9B (alternative):**
```bash
huggingface-cli download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir ./models
```

**Gemma 4 26B-A4B MoE (recommended alternative for macOS 32GB+):**
```bash
huggingface-cli download unsloth/gemma-4-26B-A4B-it-GGUF gemma-4-26B-A4B-it-Q4_K_M.gguf --local-dir ./models
```

**Gemma 4 31B dense (for comparison benchmarking):**
```bash
huggingface-cli download unsloth/gemma-4-31B-it-GGUF gemma-4-31B-it-Q4_K_M.gguf --local-dir ./models
```

**Alternate: Nemotron 3 Nano 4B (lighter, faster, 262K context):**
```bash
huggingface-cli download unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf --local-dir ./models
```

### 3. Start the server

**macOS (Apple Silicon):**
```bash
./llama-server \
  -m models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 \
  -c 131072 \
  -b 4096 \
  -fa on \
  -np 1 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --reasoning-budget -1 \
  --metrics
```

**Linux/WSL (NVIDIA GPU):**
```bash
./llama-server \
  -m models/gemma-4-E4B-it-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 \
  -c 131072 \
  -b 4096 \
  -fa on \
  -np 1 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --reasoning-budget -1 \
  --metrics
```

**Flag breakdown:**

| Flag | Purpose |
|------|---------|
| `-ngl 99` | Offload all layers to GPU |
| `-c 131072` | Context window (model supports up to 262K) |
| `--cache-type-k q4_0 --cache-type-v q4_0` | Quantize KV cache to fit large context in VRAM |
| `--reasoning-budget -1` | Allow thinking mode (model reasons before answering) |
| `-fa on` | Flash attention |
| `-np 1` | Single slot (saves memory) |
| `--metrics` | Enable `/metrics` endpoint |
| `-b 4096` | Batch size for prompt processing |

**For 8GB GPUs**, reduce context to avoid OOM:
```bash
-c 32768   # or -c 16384 if still OOM
```

### 4. Verify it works

```bash
# Check the web UI
open http://localhost:8080

# Test the API
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Write a hello world in Python"}],
    "max_tokens": 200
  }'
```

## Why MoE (Mixture of Experts)?

MoE models have more total parameters but only **activate a fraction per token**. This means less computation per token = faster inference, while the model retains knowledge from all its parameters.

| Model | Total Params | Active Params | Q4 Size | Context | Quality Tier |
|-------|-------------|---------------|---------|---------|-------------|
| Nemotron 3 Nano 4B | 4B | 4B (hybrid Mamba-2) | ~2.5GB | 262K | Below Haiku (edge/agent) |
| **Gemma 4 E4B** | **8B** | **~2B (MoE)** | **4.7GB** | **131K** | **Haiku+ (fast)** |
| Qwen3.5-9B | 9B | 9B (dense) | 5.3GB | 131K | GPT-4o-mini / Haiku |
| Qwen3.5-27B | 27B | 27B (dense) | 16GB | 131K | Sonnet-ish |
| **Gemma 4 26B-A4B** | **26B** | **4B (MoE)** | **16.9GB** | **256K** | **TBD** |
| Gemma 4 31B | 31B | 31B (dense) | 18.3GB | 256K | TBD |
| Qwen3.5-35B-A3B | 35B | 3B (MoE) | 22GB | 131K | Sonnet 4.5 |
| **Qwen3.6-35B-A3B** | **35B** | **3B (MoE)** | **22GB (Q4)** | **262K (1M ext.)** | **Sonnet 4.5+ (73.4% SWE-bench)** |
| **Qwen3.6-35B-A3B (12GB VRAM)** | **35B** | **3B (MoE)** | **11.5GB (IQ2)** | **57K** | **Sonnet-class** |

The 35B-A3B is the sweet spot for Apple Silicon: it's **faster than the 9B** (29 vs 20 tok/s on M3 Pro) because it only computes 3B params per token, yet **smarter than the 27B** because it draws from 35B total parameters. Qwen 3.5 35B-A3B [beat Sonnet 4.5 on several benchmarks](https://venturebeat.com/technology/alibabas-new-open-source-qwen3-5-medium-models-offer-sonnet-4-5-performance); the **Qwen 3.6 35B-A3B** (April 16, 2026) pushes further — 73.4% SWE-bench Verified, 37.0% MCPMark (tool use), and native 262K context extensible to ~1M with YaRN.

**The tradeoff:** MoE models are larger on disk (22GB vs 16GB for the dense 27B) because they store all expert weights even though only a subset is used per token. On machines with limited RAM, the 9B dense model may be the better fit.

### Thinking mode

These models support a "thinking" mode where they reason through problems before answering — similar to Claude's extended thinking or DeepSeek R1. This is controlled by `--reasoning-budget`:

| Flag | Behavior |
|------|----------|
| `--reasoning-budget -1` | Thinking enabled (recommended) — model decides when to think |
| `--reasoning-budget 0` | Thinking disabled — faster but lower quality on hard tasks |
| `--reasoning-budget 1024` | Cap thinking at 1024 tokens |

We recommend `-1` (unlimited) for coding tasks. The model only thinks when it determines the problem is complex enough to warrant it, so simple requests stay fast.

## Inference Engines (NVIDIA CUDA)

llama.cpp is the reliable default. ExLlamaV3 + TabbyAPI is ~50-60% faster using custom CUDA kernels but only supports EXL3 format.

| Engine | ~Tok/s (9B, 4-bit) | Format | Gemma 4? | Setup |
|--------|-------------------|--------|----------|-------|
| **ExLlamaV3 + TabbyAPI** | **~100-130** | EXL3 (CUDA-only) | Dense yes; **E4B no** (as of 2026-07) | `git clone` + `start.sh` |
| **TensorRT-LLM** | **~80-95** | TRT engine / HF | Not yet (Gemma 3 only) | Docker container |
| **llama.cpp** | ~65-94* | GGUF (universal) | Yes (day-0) | Build or download binary |
| Ollama | ~62 | GGUF (llama.cpp backend) | Yes | `brew install ollama` |

\*94 tok/s with Gemma 4 E4B (MoE, ~2B active), 65-78 tok/s with Qwen3.5-9B (dense).

**Why two formats?** GGUF is universal (CUDA, Metal, CPU, Vulkan) using generic compute kernels. EXL3 trades portability for speed — it uses hand-tuned CUDA kernels optimized for NVIDIA memory hierarchy, so it only runs on NVIDIA GPUs but is significantly faster.

### ExLlamaV3 + TabbyAPI (fastest for NVIDIA)

ExLlamaV3 uses the EXL3 format with calibration-based quantization — it measures which layers matter most and allocates more bits to them. TabbyAPI serves it with an OpenAI-compatible API.

> **Note (updated 2026-07):** ExLlamaV2 is archived; ExLlamaV3 + EXL3 is the active project. ExLlamaV3 **now supports dense Gemma 4** (`Gemma4ForConditionalGeneration`) and Qwen 3.5 (incl. MoE) — but **not the E2B/E4B variants**, and **Qwen 3.6 is not yet listed**. So for the guide's Gemma 4 **E4B** speed pick, still use llama.cpp on CUDA; EXL3 is for dense Gemma 4 / Qwen 3.5.

**Setup:**
```bash
git clone https://github.com/theroyallab/tabbyAPI
cd tabbyAPI

# Linux/WSL
./start.sh

# Windows
start.bat
```

First launch is slow (JIT-compiles CUDA extensions). Subsequent launches are fast.

**Download a model:**
```bash
# Qwen3.5-9B at 4.0 bits per weight (~4.5GB, fits easily in 12GB VRAM)
./start.sh download turboderp/Qwen3.5-9B-exl3 --revision 4.00bpw

# Qwen3.5-9B at 5.0 bpw (~5.6GB, higher quality, still fits 12GB)
./start.sh download turboderp/Qwen3.5-9B-exl3 --revision 5.00bpw
```

**Configure** `config.yml`:
```yaml
host: 0.0.0.0
port: 5000
disable_auth: true

backend: exllamav3
model_dir: models
model_name: Qwen3.5-9B-exl3

max_seq_len: 32768
cache_size: 32768
cache_mode: Q4

gpu_split_auto: true
reasoning: true
```

**Claude Code integration:** TabbyAPI is OpenAI-only, so Claude Code needs a LiteLLM proxy in front (see the [Claude Code note](#claude-code-recommended-for-agentic-coding) — the `openai/` prefix goes in the LiteLLM config, not here). Point Claude Code at the proxy:
```bash
# after starting LiteLLM (see the Cursor section) mapping a model_name -> openai/<tabby-model>
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_AUTH_TOKEN=local \
claude --model qwen
```

**Or use the script:**
```bash
./start-tabby.sh              # default: Qwen3.5-9B 4.0bpw
./start-tabby.sh 5bpw         # higher quality 5.0bpw
```

### EXL3 models for 12GB VRAM

| Model | Repo | bpw | Size | Fits 12GB? |
|-------|------|-----|------|------------|
| **Qwen3.5-9B** | `turboderp/Qwen3.5-9B-exl3` | 4.0 | ~4.5GB | Yes (recommended) |
| Qwen3.5-9B | `turboderp/Qwen3.5-9B-exl3` | 5.0 | ~5.6GB | Yes (higher quality) |
| Qwen3.5-9B | `turboderp/Qwen3.5-9B-exl3` | 6.0 | ~6.7GB | Yes |
| Gemma 4 dense (26B/31B) | search HF `exl3` | 3–4 | varies | Dense yes; **E4B not supported** |

\*As of 2026-07, ExLlamaV3 supports **dense Gemma 4** and Qwen 3.5 (incl. MoE), but **not the E2B/E4B** variants, and Qwen 3.6 is not yet listed. For the E4B speed pick use llama.cpp. Watch the [ExLlamaV3 repo](https://github.com/turboderp-org/exllamav3) for E4B/Qwen 3.6 support.

### TensorRT-LLM (maximum optimization after you've chosen a model)

Once you've benchmarked models with llama.cpp / ExLlamaV3 and picked one, TensorRT-LLM compiles it into a GPU-specific execution graph optimized for your exact hardware. It's the most work to set up but produces the most optimized inference.

> **Current limitations:** Qwen3.5 only via AutoDeploy (beta) — Qwen3 is fully supported. Gemma 4 not supported yet (Gemma 3 only). FP4 requires Blackwell GPUs — Ada (4070 Ti) maxes out at FP8/INT4.

**Install via Docker (recommended):**
```bash
# Verify GPU in WSL2
nvidia-smi

# Launch TRT-LLM container
docker run --rm -it \
  --ipc host --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10
```

**Option A: PyTorch backend (no build step, fast start):**
```bash
# Serve directly from HuggingFace — downloads and starts in 1-3 min
trtllm-serve Qwen/Qwen3-8B \
  --backend pytorch \
  --host 0.0.0.0 --port 8000 \
  --max_batch_size 1 \
  --max_seq_len 32768 \
  --kv_cache_dtype fp8 \
  --kv_cache_free_gpu_memory_fraction 0.85

# Or use NVIDIA's pre-quantized FP8 checkpoint
trtllm-serve nvidia/Qwen3-8B-FP8 \
  --backend pytorch \
  --host 0.0.0.0 --port 8000 \
  --max_batch_size 1 \
  --max_seq_len 32768 \
  --kv_cache_dtype fp8 \
  --kv_cache_free_gpu_memory_fraction 0.85
```

**Option B: TensorRT engine build (maximum performance):**
```bash
# Step 1: Quantize to INT4 AWQ (use --device cpu if 12GB OOMs)
python quantization/quantize.py \
  --model_dir ./models/Qwen3-8B \
  --dtype float16 \
  --qformat int4_awq \
  --awq_block_size 128 \
  --output_dir ./checkpoints/qwen3-8b-int4-awq \
  --calib_size 32

# Step 2: Build engine (10-30 min, writes GPU-specific binary)
trtllm-build \
  --checkpoint_dir ./checkpoints/qwen3-8b-int4-awq \
  --output_dir ./engines/qwen3-8b-int4-awq \
  --gemm_plugin float16

# Step 3: Serve the compiled engine (~90s to load on subsequent starts)
trtllm-serve ./engines/qwen3-8b-int4-awq \
  --backend tensorrt \
  --tokenizer Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 8000
```

**Quantization options on Ada Lovelace (RTX 4070 Ti):**

| Method | Size (8B model) | Speed | Quality | Ada Support |
|--------|-----------------|-------|---------|-------------|
| FP8 (recommended) | ~8GB | Fastest | Best | Yes |
| INT4 AWQ (W4A16) | ~4.5GB | Fast | Good | Yes |
| W4A8 AWQ | ~4.5GB | Faster | Good | Yes |
| FP4 / NVFP4 | — | — | — | **No (Blackwell only)** |

**Claude Code integration:** TensorRT-LLM's `trtllm-serve` is OpenAI-only, so Claude Code needs a LiteLLM proxy in front (see the [Claude Code note](#claude-code-recommended-for-agentic-coding)). Point Claude Code at the proxy, not at `:8000` directly:
```bash
# after starting LiteLLM (see the Cursor section) mapping a model_name -> openai/<trt-model>
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_AUTH_TOKEN=local \
claude --model qwen
```

**Performance vs other engines (estimated, 8-9B INT4, RTX 4070 Ti):**

| Engine | ~Tok/s | Notes |
|--------|--------|-------|
| ExLlamaV3 (EXL3) | ~100-130 | Custom CUDA kernels, single-user champion |
| **llama.cpp (Gemma 4 E4B)** | **~94** | **MoE model closes the gap with EXL3** |
| TensorRT-LLM (TRT engine) | ~80-95 | GPU-specific compiled graph, higher VRAM overhead |
| TensorRT-LLM (PyTorch) | ~75-85 | No build step, good middle ground |
| llama.cpp (Qwen3.5-9B) | ~65-78 | Dense model, broadest compatibility |

TRT-LLM's advantage grows with batched/concurrent inference. For single-user coding, ExLlamaV3 may match or exceed it due to its hand-tuned decode kernels. TRT-LLM becomes the clear winner when serving multiple users or running longer-context workloads where its compiled attention kernels shine.

## Inference Engines (macOS Apple Silicon)

**Ollama (MLX) is recommended for macOS** for its ease and correct handling of Gemma 4's thinking tokens — but it is *not* the fastest. In testing (M3 Pro 36GB, 2026-07-24), raw **`mlx-lm` decodes ~30–40% faster** than Ollama for the same model (Qwen 3.6 35B-A3B: ~48 vs ~35 tok/s), because Ollama's `q4_0` KV-cache + serving layer adds overhead. Use Ollama for convenience, `mlx-lm` for maximum single-stream decode.

| Engine | Decode Speed | Prefill Speed | Format | Claude Code API | Setup |
|--------|-------------|---------------|--------|-----------------|-------|
| **mlx-lm** | **~48 tok/s (Qwen 3.6, tested)** | high | MLX | OpenAI — needs LiteLLM proxy | `pip install mlx-lm` |
| **Ollama 0.32+** | **~33 tok/s (tested)** | **~145–244 tok/s** | Auto (MLX) | **Native Anthropic** (0.14+, no proxy) | `brew install ollama` |
| llama.cpp | ~29 tok/s (Qwen) | ~70 tok/s | GGUF | **Native Anthropic** (since Jan 2026, PR #17570) | `brew install llama.cpp` |
| vllm-mlx | MLX-native, ~400 tok/s batched | Good | MLX | **Native Anthropic** (no proxy) | `pip install` from git |
| **vLLM Metal v0.3.0-dev** | **MLX-native, batched** | **83× v0.1 TTFT** | MLX | OpenAI | Docker Desktop 4.62+ |

> **Note:** The old llama.cpp Gemma 4 thinking-token bug ([#21321](https://github.com/ggml-org/llama.cpp/issues/21321)) was **fixed 2026-04-07** (PR #21566) and was actually a CUDA/ROCm fusion bug, not a Metal issue — Gemma 4 runs fine in llama.cpp on Mac now. Ollama remains the easy path, not a required workaround.

### Ollama (recommended for macOS)

Ollama 0.20+ uses Apple's MLX framework on Apple Silicon. It handles Gemma 4's thinking tokens correctly and delivers ~31 tok/s generation / ~99 tok/s prefill on M3 Pro 36GB.

```bash
# Install
brew install ollama

# Start the server
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q4_0 ollama serve &

# Pull and run Gemma 4 26B-A4B (~17GB download)
ollama pull gemma4:26b-a4b-it-q4_K_M
ollama run gemma4:26b-a4b-it-q4_K_M

# Or use the script:
./start-ollama-mac.sh              # default: 26b-a4b
./start-ollama-mac.sh 31b          # dense 31B
./start-ollama-mac.sh e4b          # lightweight 4.5B
```

Claude Code integration:
```bash
ANTHROPIC_BASE_URL=http://localhost:11434 \
ANTHROPIC_AUTH_TOKEN=local \
claude --model gemma4:26b-a4b-it-q4_K_M
```

### vllm-mlx (recommended for Claude Code)

vllm-mlx exposes a native Anthropic `/v1/messages` endpoint — no LiteLLM proxy or `openai/` prefix needed.

```bash
# Install
pip install git+https://github.com/waybarrios/vllm-mlx.git

# Serve Gemma 4 26B-A4B
vllm-mlx serve mlx-community/gemma-4-26B-A4B-it-4bit --port 8000

# Or use the script:
./start-vllm-mlx-mac.sh            # default: 26b-a4b
./start-vllm-mlx-mac.sh 31b        # dense 31B
./start-vllm-mlx-mac.sh e4b        # lightweight 4.5B
```

Claude Code integration (cleanest — native Anthropic API):
```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=local \
claude
```

### vLLM Metal (Docker Model Runner, April 2026)

[vLLM Metal](https://github.com/vllm-project/vllm-metal) is the official vLLM hardware plugin for Apple Silicon — separate project from `vllm-mlx`. v0.2.0 (April 2026) introduced a unified paged varlen Metal kernel as the default attention backend, reporting **83× TTFT** and **3.6× throughput** vs v0.1.0.

Shipping path is via Docker Model Runner on Docker Desktop 4.62+:

```bash
# Pull an MLX-format model via Docker Model Runner
docker model pull mlx-community/Qwen3.6-35B-A3B-4bit
docker model run mlx-community/Qwen3.6-35B-A3B-4bit
```

OpenAI-compatible endpoint exposed on the local Docker model runtime — point Claude Code at it the same way as Ollama.

> **vllm-mlx vs vLLM Metal:** `vllm-mlx` (waybarrios) is a community fork with native Anthropic API support, MCP tool calling, and SSD-tiered KV cache. `vllm-metal` is the official vLLM plugin via Docker. Pick `vllm-mlx` for Claude Code without a proxy; pick `vLLM Metal` if you already use Docker and want the official path.

## Supported Flows

```
Flow 1: Claude Code (local)
  Claude Code -> localhost:8080 -> llama-server (Qwen3.5)
  Claude Code -> localhost:11434 -> Ollama MLX (Gemma 4)
  No extras needed. Just env vars.

Flow 2: Claude Code (remote, e.g. from MacBook)
  MacBook Claude Code -> cloudflared tunnel -> llama-server (on PC)
  Needs cloudflared on the PC.

Flow 3: Cursor (Chat/Cmd+K only — Agent mode unsupported)
  Cursor -> Cursor servers -> cloudflared -> LiteLLM (name mapping) -> llama-server
  Needs LiteLLM + cloudflared.
```

### Scripts

| Script | Flow | Platform | What it does |
|--------|------|----------|-------------|
| `start-server.sh` | All | Linux/WSL | Start llama-server (`9b`, `nemotron`, `gemma4-26b`, `gemma4-31b`, `gemma4-e4b`) |
| `start-server-mac.sh` | All | macOS | Start llama-server (`9b`, `35b-a3b`, `27b`, `gemma4-26b`, `gemma4-31b`, `gemma4-e4b`) |
| `start-tabby.sh` | 1 | Linux/WSL | Start TabbyAPI + ExLlamaV3 (`4bpw`, `5bpw`, `6bpw`) |
| `start-ollama-mac.sh` | 1 | macOS | Start Ollama with Gemma 4 (`26b-a4b`, `31b`, or `e4b`) |
| `start-vllm-mlx-mac.sh` | 1 | macOS | Start vllm-mlx with Gemma 4 (native Anthropic API) |
| `start-claude-local.sh` | 1 | Any | Launch Claude Code with local model (auto-detects Ollama) |
| `start-claude-windows.sh` | 1 | Linux/WSL | **Claude Code full stack: llama-server (`-ncmoe` headroom) + LiteLLM + `claude`** |
| `start-remote.sh` | 2 | Linux/WSL | Tunnel llama-server for remote access (**public internet** — prefer the LAN path below) |
| `connect-lan-mac.sh` | 2 | macOS/Linux | **Connect Claude Code to llama-server on another LAN machine** (API key + preflight check) |
| `windows/serve-rpc.ps1` | 3 | Windows | **Distributed inference** — split one model across this GPU + a remote `rpc-server` peer |
| `windows/serve-q4-solo.ps1` | 3 | Windows | Q4_K_M on the GPU **alone** via high `-ncmoe` — the control RPC must beat |
| `windows/claude-local.ps1` | 1 | Windows | **Claude Code against the local model** — starts llama-server if needed |
| `windows/claude-cloud.ps1` | — | Windows | Claude Code against the Anthropic cloud (clears `ANTHROPIC_*` so it can't be hijacked) |
| `windows/serve-lan.ps1` | 2 | Windows | Serve the model to your LAN (`--api-key-file`, binds `0.0.0.0`) |
| `windows/setup-lan-firewall.ps1` | 2 | Windows | One-time inbound rule, **LocalSubnet only** (run elevated) |
| `windows/rotate-keys.ps1` | 2 | Windows | List valid keys (`-Show`), or restart to apply revocations |
| `windows/make-shortcuts.ps1` | — | Windows | Desktop shortcuts for the local and cloud launchers |
| `start-cursor-local.sh` | 3 | Linux/WSL | LiteLLM proxy + tunnel for Cursor |
| `bench-claude-code.sh` | 1 | macOS | Benchmark Claude Code against local models (see [Benchmarking](#benchmarking-claude-code-harness)) |
| `stop-all.sh` | — | Any | Kill everything |

Always run `start-server.sh` or `start-server-mac.sh` first, then pick your flow.

## IDE Integration

### Claude Code (recommended for agentic coding)

> **How Claude Code talks to a local model:** Claude Code speaks **only the Anthropic Messages API** (`/v1/messages`) — it is *not* an OpenAI client. That splits local servers into two cases:
> - **Native Anthropic — connect directly, no proxy, no prefix.** **Ollama 0.14+** and **vllm-mlx** expose `/v1/messages`. Point Claude Code straight at them with the plain model name (note: base URL has **no `/v1`**):
>   `ANTHROPIC_BASE_URL=http://localhost:11434 ANTHROPIC_AUTH_TOKEN=local claude --model qwen3.6:35b`
> - **OpenAI-only — needs a LiteLLM proxy.** **TabbyAPI and TensorRT-LLM** expose only OpenAI `/v1/chat/completions`, which Claude Code cannot consume directly. Run a [LiteLLM proxy](#cursor-limited--agent-mode-unsupported) to translate Anthropic↔OpenAI, then point Claude Code at the proxy.
>
> **Update (2026-08):** **llama.cpp no longer belongs in the proxy list.** `llama-server` gained a native Anthropic Messages API in [PR #17570](https://github.com/ggml-org/llama.cpp/pull/17570) (January 2026) — `POST /v1/messages` plus `/v1/messages/count_tokens`, with streaming, tool use, vision, and extended thinking. Verified present in the Homebrew build here (`b9960`). Point Claude Code straight at `llama-server` with no proxy:
> `ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=local claude --model <name>`
>
> The `openai/<model>` prefix is a **LiteLLM routing convention** (it tells LiteLLM the backend is OpenAI-format) — it belongs in the LiteLLM config, never in a `claude --model` that points straight at Ollama or vllm-mlx. On a Mac, **Ollama is the zero-proxy path** and is what the rest of this section assumes.

#### Windows + NVIDIA — native, no WSL, no proxy (validated 2026-09-08)

> **This section was rewritten 2026-09-08.** It previously said *"llama.cpp is OpenAI-only,
> so the Windows path is llama-server → LiteLLM → Claude Code."* **That is no longer true.**
> llama.cpp [PR #17570](https://github.com/ggml-org/llama.cpp/pull/17570) (Jan 2026) added a
> native Anthropic Messages API at `/v1/messages`, so **LiteLLM is no longer needed**:
>
> ```
> before:  llama-server → LiteLLM(:4000) → Claude Code
> after:   llama-server(:8080) → Claude Code
> ```
>
> Tool use requires `--jinja`. The old WSL2 + LiteLLM path still works and is kept below,
> but the native path is simpler and faster. `start-claude-windows.sh` is the legacy script.

Claude Code also [runs natively on Windows](https://code.claude.com/docs/en/setup) now
(PowerShell or CMD, no WSL, no Node):

```powershell
irm https://claude.ai/install.ps1 | iex
```

Prebuilt CUDA binaries mean no compile — grab `llama-<build>-bin-win-cuda-13.3-x64.zip`
plus `cudart-llama-bin-win-cuda-13.3-x64.zip` from
[llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases), unzip both to the
same folder, then:

```powershell
windows\claude-local.ps1                       # starts llama-server if needed, then Claude Code
windows\claude-local.ps1 -Ncmoe 8 -Ctx 65536   # the measured optimum (see tuning table)
```

Or drive it yourself:

```powershell
llama-server.exe -m <model>.gguf --host 127.0.0.1 --port 8080 `
  -ngl 99 -ncmoe 8 -c 65536 -np 1 -fa on `
  --cache-type-k q4_0 --cache-type-v q4_0 `
  --jinja --reasoning-budget 0 -a qwen3.6-local

$env:ANTHROPIC_BASE_URL='http://localhost:8080'
$env:ANTHROPIC_AUTH_TOKEN='local'
claude --model qwen3.6-local --disallowedTools WebSearch WebFetch
```

> **`-a <alias>` is load-bearing.** Without it the model isn't addressable by name and
> `claude --model` won't resolve.
>
> **Watch for a stray `llama-server.exe`.** Recent builds default to a *router* mode, so an
> argument-less `llama-server.exe` will happily bind :8080 and answer every request with
> `400: model '<name>' not found`. If you see that error, check for a second process.


> **Disable WebSearch/WebFetch.** These are Anthropic **server-side** tools (their tool `type` is `web_search_*`/`web_fetch_*`, not `function`), so llama-server rejects them with `400: 'type' of tool must be 'function'` — and they can't run against a local model anyway. `--disallowedTools WebSearch WebFetch` removes them; `start-claude-windows.sh` does this automatically.

> **The 12GB gotcha — you must offload some experts to CPU.** At full GPU offload the Qwen3.6-35B-A3B IQ2 pick uses **~all 12GB (free=0)** and is unstable for agentic use: it loads, then OOM-exits the moment Claude Code's **~25K-token** system prompt + tools (or a CUDA graph) needs more VRAM. The fix is `--n-cpu-moe` (`-ncmoe`): keep some MoE expert layers on CPU. It's also your main **speed** dial — measured on a 4070 Ti at 128K context:
>
> | `-ncmoe` | VRAM | Decode | Notes |
> |---------:|-----:|-------:|-------|
> | 16 | 8.8GB | ~40 tok/s | most headroom |
> | 8 | 10.6GB | **~77 tok/s** | ~2× faster, still fits |
> | 0 (full GPU) | >12GB | — | OOM at large context |
>
> Fewer CPU experts = faster decode but less VRAM. `start-claude-windows.sh` defaults to `NCMOE=16`, `CTX=40960` (enough for Claude Code's ~25K prompt), and `REASONING=0` (thinking off — fastest, best for the tool loop; set `REASONING=medium` for a capped 1536-token thinking budget, or `-1` for unlimited).

> **Does it actually work?** Yes — validated on this RTX 4070 Ti: pointed at a small project with a failing test, the local model drove the full agent loop (read → edit → run `pytest` → report) and fixed the bug with correct, documented code. Tool-calling held up on the 2-bit quant. Expect it to be slow and to need the `-ncmoe` headroom above; it is genuinely usable for the daily small-edit 70–80%, not the hard 20%.

##### Tuning for a single agent (measured 2026-09-08, RTX 4070 Ti 12GB)

All at `-ncmoe 8`, `-np 1`, Qwen3.6-35B-A3B IQ2_M, `q4_0` KV cache. Decode measured
over a 300-token generation; VRAM is total card usage including ~2.4GB of desktop.

| `-c` (context) | VRAM used | Free | Decode |
|---------------:|----------:|-----:|-------:|
| 40,960 | 10273 MB | 2009 MB | 99.4 tok/s |
| **65,536** | **10783 MB** | **1499 MB** | **100.5 tok/s** |
| 98,304 | 11279 MB | 1003 MB | **48.2 tok/s** |
| 131,072 | 11833 MB | 449 MB | unsafe — OOM risk under load |

> **64K is the knee, and it is sharp.** Going from 64K to 96K *halves* decode speed
> while buying context you will rarely use — the KV cache crowds the card and the
> whole thing slows down. Context costs **speed**, not just memory, which is the part
> that surprises people. Below 64K you gain nothing: 40K and 64K decode identically.
>
> **`-ncmoe` is the other dial.** At `-ncmoe 16` (the old default) decode was ~40 tok/s;
> `-ncmoe 8` gives **~100 tok/s** — a 2.5× win — and still leaves 1.5GB free at 64K.
> Fewer CPU-resident experts is faster *and* uses more VRAM, so it trades against `-c`.
>
> **Claude Code's 1M-token window is irrelevant locally.** At ~650 tok/s prefill, a
> 1M-token context would spend **~26 minutes** prefilling before the first token. Even
> 128K is ~3.4 minutes cold. The binding constraint is time, not memory — which is why
> the recommended default is 64K, not "as much as fits".

**Recommended single-agent defaults** (now baked into `serve-lan.ps1` and `claude-local.ps1`):

```
-ngl 99 -ncmoe 8 -c 65536 -np 1 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--jinja --reasoning-budget 0 -a qwen3.6-local
```

`--reasoning-budget 0` matters as much as the rest: thinking tokens re-prefill every
turn, so leaving it on is a large, invisible tax on an agentic loop.

#### LAN-only access (serve the Windows box to your Mac)

If the client is another machine on your own network, do **not** use the Cloudflare
tunnel in `start-remote.sh` — that publishes your GPU to the public internet. Serve it
on the LAN instead: lower latency, no third party, far smaller exposure.

`llama-server` speaks the Anthropic Messages API natively, so the whole path is just
**llama-server → Claude Code across the network**. No LiteLLM, no tunnel.

**On the serving machine (Windows):**

```powershell
# once, from an ELEVATED PowerShell -- opens the port to the local subnet only
J:\llama\setup-lan-firewall.ps1

# each time
J:\llama\serve-lan.ps1
```

`serve-lan.ps1` binds `0.0.0.0`, requires one of the keys in `J:\llama\api-keys.txt`, and
prints the exact client command with your LAN IP filled in.

**Key changes don't need a server restart.** `--api-key-file` accepts many keys, one per
line, and *every* listed key is valid at once — so pre-provision a few spares and adding
a device or switching keys costs nothing:

```
# in use -- Mac laptop
correct-horse-battery-staple

# spares -- already valid, just start using one
another-four-word-key
```

> **Verified:** the file is read at **startup only**. Appending a key returns `401` until
> the server restarts — there is no reload endpoint and no SIGHUP handler. That is exactly
> why you pre-provision spares. *Revoking* a key still needs a restart (`rotate-keys.ps1`,
> ~40s model reload); *adding* one never does.

`rotate-keys.ps1 -Show` lists the valid keys; without `-Show` it restarts to apply deletions.

Per-device keys are the useful side effect — revoke one machine without disturbing the rest.

**On the client (Mac/Linux)** — configure once:

```bash
cat > ~/.llama-lan.conf <<'CONF'
LLAMA_HOST=192.168.5.195
LLAMA_KEY=llk_your_key_here
LLAMA_PORT=8080
LLAMA_MODEL=qwen3.6-local
CONF
chmod 600 ~/.llama-lan.conf
```

Then every time, with no arguments and nothing to edit:

```bash
./connect-lan-mac.sh                      # just runs Claude Code
./connect-lan-mac.sh -p "fix the test"    # extra args pass through
```

Config resolves from environment → `./.llama-lan.conf` → `~/.llama-lan.conf`, so you can
override per-project. `.llama-lan.conf` is gitignored — the key never enters the repo.

The script verifies reachability *and* the key before launching Claude Code, so a firewall
or subnet problem gives you a clear error instead of a silent retry loop.

**Two independent controls keep this local:**

| Control | Effect |
|---|---|
| `--api-key-file` on the server | Requests without a valid bearer token are rejected — a machine that can reach the port still can't use it |
| `-RemoteAddress LocalSubnet` on the firewall rule | Only hosts in your own subnet can open the port at all |

> **Binding `0.0.0.0` does not expose you to the internet** on its own — your router
> won't forward the port unless you add a port-forward or DMZ entry. Don't.
>
> **Traffic is plain HTTP**, so prompts and code cross the network unencrypted. Fine on
> a home LAN you control; not fine on shared or public Wi-Fi. `llama-server` accepts
> `--ssl-key-file`/`--ssl-cert-file` if you want TLS.
>
> **Check your network category.** If Windows classifies the interface as *Public*
> (`Get-NetConnectionProfile`), it applies stricter defaults. The `LocalSubnet` scope
> still contains the rule, but on a genuinely untrusted network "local subnet" means
> everyone else on it — only do this on a network you own.

**Flow 1: Local (same machine — macOS/Ollama)**

```bash
# Terminal 1: start the server
./start-server-mac.sh   # macOS
./start-server.sh       # Linux/WSL

# Terminal 2: Claude Code with local Qwen
./start-claude-local.sh
```

Or manually — **Ollama (native Anthropic, no proxy, no prefix):**
```bash
ANTHROPIC_BASE_URL=http://localhost:11434 ANTHROPIC_AUTH_TOKEN=local claude --model qwen3.6:35b
```
> `start-server-mac.sh`/`start-server.sh` launch `llama-server`, which is OpenAI-only — to drive Claude Code from *that*, put a LiteLLM proxy in front (see the note above). For a no-proxy local setup, run Ollama (`./start-ollama-mac.sh`) and use the command above.

You can run both side by side — normal `claude` for complex tasks, local Qwen for quick edits.

**Flow 2: Remote (e.g. from MacBook)**

On your PC:
```bash
# Terminal 1
./start-server.sh

# Terminal 2
./start-remote.sh
# Prints a tunnel URL like https://xxx-xxx.trycloudflare.com
```

On your MacBook:
```bash
ANTHROPIC_BASE_URL=https://xxx-xxx.trycloudflare.com \
ANTHROPIC_AUTH_TOKEN=local \
claude --model qwen3.6:35b
```

### Cursor (limited — Agent mode unsupported)

> **Known limitation (as of March 2026):** Cursor's Agent mode routes all requests through Cursor's servers and does not support custom API keys/endpoints. Custom models only work in **Chat** and **Cmd+K** modes. Cursor also has client-side model name validation that rejects most custom model names.

**Workaround using LiteLLM proxy:**

1. Install LiteLLM:
   ```bash
   pip install 'litellm[proxy]'
   ```

2. Create `litellm-config.yaml`:
   ```yaml
   model_list:
     - model_name: deepseek-r1-0528
       litellm_params:
         model: openai/deepseek-r1-0528
         api_base: http://localhost:8080/v1
         api_key: local
   ```

3. Start the proxy:
   ```bash
   litellm --config litellm-config.yaml --port 4000 --host 0.0.0.0
   ```

4. Cursor needs HTTPS, so tunnel it:
   ```bash
   cloudflared tunnel --url http://localhost:4000 --protocol http2
   ```

5. In Cursor Settings -> Models:
   - Click **+ Add Custom Model** -> `deepseek-r1-0528`
   - Override OpenAI Base URL -> `https://<your-tunnel>.trycloudflare.com/v1`
   - API Key -> any string (e.g. `sk-1234`)

6. Select `deepseek-r1-0528` in Chat mode (not Agent mode).

We use `deepseek-r1-0528` as the model name because Cursor's validation accepts it. LiteLLM maps it to your local Qwen server.

### Cline (VS Code extension — full agent mode)

Cline supports local models with no restrictions:

1. Install [Cline](https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev) in VS Code
2. Settings -> API Provider -> OpenAI Compatible
3. Base URL: `http://localhost:8080/v1`
4. Model: `qwen`
5. API Key: any string

Works in agent mode — no tunnel, no name hacks.

### Continue (VS Code extension — autocomplete + chat)

Best for tab completion with local models:

1. Install [Continue](https://marketplace.visualstudio.com/items?itemName=Continue.continue) in VS Code
2. Configure `~/.continue/config.yaml` to point at `http://localhost:8080`

### Void (open source Cursor alternative)

[Void](https://voideditor.com/) is a VS Code fork with native local model support. No tunnels or workarounds needed.

## Model selection: current picks

> **This is the deliberately volatile part of the guide.** Everything here is dated and expected to churn. The stable logic is in [How to choose](#how-to-choose-start-here); this section only records which model satisfies each rule *right now*, plus what's on deck to replace it. Always confirm fit and speed on your own hardware with [`./benchmark.sh`](#benchmark-it-yourself).

### Current picks
**As of 2026-07-25.** Quality = biggest MoE that fits at ≥IQ2 with usable context; Speed = small MoE at Q4. Tok/s and VRAM come from [our benchmark run](#performance) / your own `./benchmark.sh` — they are not restated here so they can't rot in two places.

| Budget | Quality pick | Speed pick | Notes |
|--------|-------------|-----------|-------|
| **8GB VRAM** | Gemma 4 E4B (Q4) | Gemma 4 E4B (Q4) | one model wins both at this tier |
| **12GB VRAM** | Qwen3.6-35B-A3B (UD-IQ2_M) | Gemma 4 E4B (Q4) | quality pick is 2-bit — see caveat below |
| **16GB VRAM** | Qwen3.6-35B-A3B (IQ3) | Gemma 4 E4B (Q4) | IQ3 buys quality headroom + longer context |
| **24GB VRAM** | Qwen3.5-27B (Q4) or A3B MoE (Q4) | Gemma 4 E4B / A3B (Q4) | full-Q4 MoE fits here |
| **32–36GB Apple** | Qwen3.6-35B-A3B (Q4) | Gemma 4 26B-A4B (Q4) | see [macOS engines](#inference-engines-macos-apple-silicon); avoid dense 31B (swap-bound) |

**Caveat on the 12GB quality pick:** the headline "73.4% SWE-bench / Sonnet-class" figure for Qwen3.6-35B-A3B is the *full-precision* model. The 12GB pick runs it at **UD-IQ2_M (~2-bit)**, which degrades quality by an unmeasured amount — treat it as "Sonnet-class *architecture*, quantization-reduced," not a guaranteed Sonnet-class result. This is exactly why the dense 16B challenger on the watchlist is worth benchmarking against it.

### Watchlist (candidates to replace the picks above)
- **DeepSeek-Coder V3 Distilled 16B** — dense 16B, ~40.5% SWE-bench Verified, fits 12GB at Q4 with *no IQ2 degradation*. The strongest honest quality-per-GB challenger to the Qwen3.6-IQ2 pick; benchmark them head-to-head on your card before committing.
- **Qwen3-Coder-Next** (80B-A3B MoE, Feb 2026) — ~70% SWE-bench, but needs ~24GB (Q4 ~52GB / 24GB VRAM minimum). The 24GB-tier aspiration, not a 12GB fit.
- **Nemotron 3 Nano 4B** — hybrid Mamba-2 (mostly Mamba-2 + 4 attention layers), ~2.5GB, 262K context. Below-Haiku coding quality — designed for edge agents. Use only when you need maximum speed or the long window. `./start-server.sh nemotron`.

_Snapshot only. Check a live leaderboard (SWE-bench Verified, [llm-stats.com](https://llm-stats.com/)) before trusting any name in this section._

### August 2026 dense models: benchmark winners, agent-loop losers

Qwen3.8-27B and Muse Glimmer 30B both beat Gemma 4 26B-A4B on published coding benchmarks
(Qwen3.8-27B: SWE-bench Pro 61.7, Terminal Bench 2.1 73.0, LiveCodeBench v6 90.3; Artificial
Analysis scores it 52 vs 38 for Qwen3.6-27B). Neither could drive Claude Code on a 36GB M3 Pro —
[measured, not assumed](#results--m3-pro-36gb-measured-2026-08-25). Benchmark scores measure what a
model knows per turn; local agentic coding is decided by how fast you can *feed* it 21k tokens of
system prompt, every turn. Keep using the MoE.

### Why not the dense 27B on Apple Silicon?

On a 36GB M3 Pro, the dense 27B model uses ~18GB for weights + 2.3GB for KV cache, leaving very little headroom. In practice this causes **swap thrashing** (~1.7 tok/s) when other apps are running. The 35B-A3B MoE is larger on disk (22GB) but faster at inference because it only activates 3B parameters per token.

### Quality tiers vs premium models

This is about *capability*, not speed — for tok/s and VRAM see [Performance](#performance) / `./benchmark.sh`. Benchmark numbers below are the models' **full-precision** published scores; a quantized local run (especially IQ2) lands somewhat lower, which is the whole point of the [12GB caveat](#current-picks).

| Model (full precision) | Published coding signal | Rough tier |
|------------------------|-------------------------|-----------|
| **Qwen3.6-35B-A3B** | 73.4% SWE-bench Verified, 51.5% Terminal-Bench 2.0, 37.0% MCPMark | Sonnet 4.5+ on coding; best local tool/MCP use |
| **Qwen3.5-35B-A3B** | competitive coding, beats Sonnet 4.5 on IFBench | Sonnet 4.5 tier |
| **Gemma 4 E4B** | comparable to 9B-dense on standard coding; ~18% MCPMark | Haiku+ (fast); weaker at multi-step & tool use |
| **Qwen3.5-9B** | solid completions/edits/boilerplate | GPT-4o-mini / Haiku |

Two honesty notes that don't go stale:
- **Quantization tax.** The 12GB quality pick runs Qwen3.6-35B-A3B at ~2-bit (UD-IQ2_M). Treat the 73.4% as an *architecture ceiling*, not what you'll get locally — measure the gap, don't assume it away.
- **The frontier still wins the hard 20%.** All local picks fall short of frontier models (Claude Opus-class) on long-horizon agentic workflows, multi-file refactors, and deep reasoning. Best strategy: route the daily 70–80% of edits to a local model, reserve a frontier API for the hard 20% (see [Hybrid Local + Cloud](#tangential-hybrid-local--cloud-agents)).

## Distributed inference (RPC) — running a model bigger than one machine

A 12GB card holds Qwen3.6-35B-A3B only at **IQ2_M** (2.7 bits/weight), where
quantisation damage starts to affect tool-call formatting. llama.cpp's RPC backend
splits one model's layers across machines, pooling memory:

```
PC (12GB VRAM)  +  Mac (36GB unified)  =  ~38GB usable  ->  up to Q6_K (29.3GB)
```

See **[MAC-RPC-SETUP.md](MAC-RPC-SETUP.md)** for the full peer setup, including the
macOS firewall specifics.

> **Try the simpler options first.** RPC pays *per-token network latency* — every
> token crosses the wire — and the model runs at the pace of its slowest shard.
> Two alternatives usually win:
>
> 1. **Q4 on the PC alone with a high `-ncmoe`.** This is a 3B-active MoE, so
>    CPU-resident experts are cheap (measured: `-ncmoe 8` gives ~100 tok/s), and
>    CPU transfers happen at PCIe speed rather than over Wi-Fi.
> 2. ~~**Q5_K_M on the Mac alone**~~ — **tested 2026-09-08 and rejected.** It loaded
>    only by evicting file cache and background apps; the machine is left one
>    page-fault storm from the swap-thrashing measured at ~6 tok/s on the dense 31B.
>    **The Mac's solo ceiling is Q4_K_M (22.1GB).**
>
> That last result cuts the other way, and is worth stating plainly: **splitting a
> larger model across both machines can be gentler than running a smaller one solo.**
> Q5 solo leaves the Mac ~8GB for macOS; Q6 split leaves it ~16GB, because the PC
> absorbs ~10GB of layers. Neither machine is maxed.
>
> The RPC backend is experimental upstream, and peers need compatible llama.cpp
> builds (there is no version negotiation). Benchmark before committing to it.

## Benchmarking (Claude Code harness)

Tok/s tells you how fast a model types, not whether it can finish a job. `bench-claude-code.sh`
drives **Claude Code itself** against a local model over Ollama's native Anthropic endpoint and
scores it on small agentic tasks that either pass or fail — no vibes, no vendor numbers.

```bash
./bench-claude-code.sh --list          # show candidate models + task list, no downloads
./bench-claude-code.sh --pull-only     # download the candidates (~20GB each)
./bench-claude-code.sh                 # full matrix
./bench-claude-code.sh --models "qwen3.8:27b-mtp-q4_K_M" --tasks "01 04"
```

### Candidate models (36GB Apple Silicon, August 2026)

| Ollama tag | Size | Why it's in the list |
|-----------|------|----------------------|
| `qwen3.8:27b-mtp-q4_K_M` | 18GB | Aug 2026 quality leader, **with MTP speculative decoding at Q4** |
| `qwen3.8:27b-q4_K_M` | 18GB | Same weights, MTP off — isolates how much MTP actually buys |
| `muse-glimmer:30b-q4_K_M-dflash` | 20GB | Meta's local-agent model + DFlash drafter bundled in |
| `gemma4:26b-a4b-it-q4_K_M` | 17GB | The incumbent from the table above — the baseline to beat. **It won: 4/4.** |

> **Qwen 3.8 makes MTP *fit*, which turned out not to be the problem.** Gemma 4's MTP was unusable
> on a 36GB Mac (bf16-only, ~62GB); Qwen 3.8 ships `qwen3.8:27b-mtp-q4_K_M` at **18GB**. It loads
> fine — and it still timed out, because MTP speeds up generation while this workload is bound by
> prompt ingestion. See [Results](#results--m3-pro-36gb-measured-2026-08-25).
>
> Both Qwen 3.8 and Muse Glimmer are **dense**, which on past evidence (the dense Gemma 4 31B at
> ~6 tok/s) is the risk to watch. That is exactly what this harness is for: if a dense 27B answers
> correctly at 12 tok/s while a MoE answers wrongly at 35 tok/s, the tok/s column was lying to you.

### Tasks

Each task is a throwaway fixture repo plus an **external verifier** that the model never sees:

| Task | What it exercises | Pass condition |
|------|-------------------|----------------|
| `01-fix-bug` | Read code, run tests, diagnose, fix | `unittest` suite green — **and** `tests/` unmodified (checksummed, so "fix the test" fails) |
| `02-refactor` | Multi-file rename across source + callers + tests | Zero occurrences of the old name, suite still green |
| `03-add-feature` | Add a CLI flag without regressing existing output | New `--json` output parses and matches; plain output byte-identical; README updated |
| `04-codebase-search` | Search/tool loop over a 30-file haystack | Correct value written to `ANSWER.txt` |

Tasks 01 and 03 are deliberately cheat-resistant — a model that edits the tests or changes the
existing output to make things pass gets a FAIL, which is the failure mode small local models hit
most often.

### Results — M3 Pro 36GB, measured 2026-08-25

```
Model                              01-fix-bug      02-refactor     03-add-feature  04-codebase     Pass  Eff tok/s
gemma4:26b-a4b-it-q4_K_M           PASS 89s        PASS 118s       PASS 195s       PASS 96s         4/4       13.0
muse-glimmer:30b-q4_K_M-dflash     FAIL 232s       FAIL 247s       FAIL 316s       (abandoned)      0/3        1.5
qwen3.8:27b-mtp-q4_K_M             TIMEOUT 905s    (not run)       (not run)       (not run)        0/1          -
```

| Model | Type | Result | What actually happened |
|-------|------|--------|------------------------|
| **Gemma 4 26B-A4B** | **MoE** | **4/4 PASS**, 8 min total | Clean: every run `is_error: false`, `stop_reason: end_turn`, all verifiers silent. Fixed the source rather than the checksummed tests, and added `--json` without disturbing existing output. |
| Muse Glimmer 30B | Dense | 0/3 | **Inert, not slow.** 1–2 turns, ~200–700 output tokens, zero tool calls. Its entire reply to the fix-the-tests prompt was *"Hi! How can I help you today?"* — it answered a greeting nobody sent. |
| Qwen 3.8 27B (MTP) | Dense | 0/1, timed out | **Slow on both axes.** ~262s one-time cold-start prefill (~82 tok/s), then ~5 tok/s decode. Gemma solved the same task in 6 turns and 89s. |

**The incumbent won, and it wasn't close.** Both August 2026 releases that beat Gemma 4 on every
published benchmark are unusable as Claude Code drivers on this hardware — for two completely
different reasons.

**The generalizable finding: the MoE beats the dense model on *both* axes, by roughly 10x on prefill
and 5x on decode.** Measured from `llama-server`'s own slot timings during these runs:

| | Prefill (prompt processing) | Decode (generation) |
|---|---|---|
| **Gemma 4 26B-A4B (MoE)** | **~720–830 tok/s** → ~26s for a 21k prompt | **~25 tok/s** |
| Qwen3.8-27B (dense) | ~82 tok/s → **~262s** for the same prompt | **~5 tok/s** |

That is what a 4B-active MoE buys you over a 27B-dense on 36GB unified memory, and it is why the
dense model timed out: **~262s of cold-start prefill, and then every turn generating at ~5 tok/s.**
It loses twice, not once.

> **Prompt caching is NOT broken — it is working.** It is tempting to blame the 21k-token prompt
> being re-sent each turn, but the server logs show Ollama's context checkpoints doing their job:
> ```
> slot create_check: task 204 | created context checkpoint 1 of 32 (n_tokens = 21316, 233 MiB)
> slot   operator(): task 248 | cached n_tokens = 21824, memory_seq_rm [21824, end)
> ```
> Turn 2 reused the cache and prefilled only the delta. **The 262s is a one-time cold-start cost per
> session, not a per-turn cost** — after that, dense models are limited by decode speed. (An earlier
> draft of this section claimed a per-turn prefill wall; the slot timings disprove it.)

> **MTP did not help.** Multi-token prediction accelerates generation, and at ~5 tok/s there was
> something to accelerate — but it did not close a 5x gap. Lowering context 65536 → 32768 changed
> prefill only marginally (87 → 82 tok/s), ruling out swap pressure as the primary cause.

**Two honest caveats.** Glimmer's failure looks like a chat-template / tool-calling integration bug
in Ollama rather than a statement about the model — it is worth retesting on a later Ollama, and it
is *not* evidence the model is bad. And Qwen 3.8 never completed a task, so its **quality is
unmeasured** here; only its cost is known. It may well be smarter per turn — it just cannot afford
the turns on this machine.

Full per-run JSON, stderr, and verifier output land in `bench/results/<timestamp>/`, and
`results.csv` has the raw rows.

### Prompt caching: what the ecosystem does, and whether it's worth your time

Everyone running a coding agent against a local model hits the same question — the client re-sends
tens of thousands of tokens every turn, so how do you avoid re-ingesting them? The state of the art
as of August 2026:

| Lever | What it does | Worth it here? |
|-------|--------------|----------------|
| **Ollama context checkpoints** | Snapshots the KV state at intelligent points and reuses shared prefixes across requests; 0.33.0 improves prefill restore points and fixes hangs when agent clients cancel long prefills — but as of 2026-08-26 it is still **`v0.33.0-rc4`** (prerelease, tagged 2026-08-21). Latest stable is v0.32.15, so Homebrew is *not* lagging | **Already on and already working** — verified in our slot logs. 0.33.0 is not in Homebrew yet (stable is 0.32.15) |
| **Long `keep_alive`** | Ollama unloads after 5 min by default, and unloading wipes the cache | **Yes — do this.** `bench-claude-code.sh` sets 60m; without it every pause costs a full cold start |
| **`CLAUDE_CODE_ATTRIBUTION_HEADER=0`** | Claude Code prepends an attribution block to the system prompt; because cache reuse depends on a *stable token prefix*, a varying header at position 0 can invalidate everything after it | **Cheap enough to just set.** One env var, confirmed present in Claude Code 2.1.245. Reported to flip llama.cpp from "full prompt re-processing" to restoring a checkpoint in ~500ms |
| **llama.cpp `--cache-reuse`, `--cache-ram`, `--slot-save-path`** | KV shifting across shared chunks, a larger host-RAM prompt cache, and save/restore of slots to disk for cross-*session* persistence | Only if you drive `llama-server` directly. Ollama covers this path already |

**Is it worth solving? Mostly no — because it is already solved, and it was never the real problem.**

The honest accounting for this machine:

- **Prefill caching is working**, so the ~21k-token ingest is a **one-time ~26s (MoE) or ~262s
  (dense 27B) cost per session**, not a per-turn tax. Turn 2 onward reuses the checkpoint.
- **The remaining cost is decode**, and no caching trick touches that. Even with prefill driven to
  zero, a dense 27B still generates at ~5 tok/s here — several hundred output tokens per agent turn
  means ~60–120s per turn regardless.
- **So prompt caching cannot rescue a dense model on 36GB, and the MoE does not need rescuing**
  (~26s cold start, ~15s/turn steady state). It changes no model-selection decision.

Do the two free things — long `keep_alive` (already in the script) and `CLAUDE_CODE_ATTRIBUTION_HEADER=0`
— and upgrade Ollama when 0.33.0 leaves release-candidate status (it was still `rc4` on 2026-08-26; Homebrew lands new stables within ~a day). Anything beyond that is optimising the half of the
problem that isn't hurting.

### Is Ollama the right harness? (engine landscape, August 2026)

Ollama did its job in these runs — native Anthropic endpoint, MLX backend, and prompt caching that
demonstrably worked. But two of our three results are arguably *engine* results rather than *model*
results, which is reason to look around:

| Engine | Anthropic `/v1/messages` | Caching / speed story | Worth testing? |
|--------|--------------------------|------------------------|----------------|
| **Ollama 0.32.15** (current) | Native (since 2026-01-16) | Context checkpoints, verified working. 0.33.0 is still at `rc4` (2026-08-21) — Homebrew tracks stable, so 0.32.15 *is* current | **Incumbent.** Easiest, unattended, menu-bar autostart |
| **llama.cpp** | **Native since Jan 2026** (PR #17570) — *not* proxy-only any more | `--cache-reuse`, `--cache-ram`, `--slot-save-path` for cross-session KV persistence | Yes, if you want fine control. Already installed here |
| **LM Studio 0.4.1+** | Native `/v1/messages` | GGUF + MLX; headless server mode (`llmster`) since Jan 2026 removed the GUI-only limitation | Maybe — GUI model browser, otherwise similar to Ollama |
| **vllm-mlx** | Native (both APIs from one process) | Continuous batching, paged KV, prefix caching, SSD-tiered cache, MCP tool calling | Yes — already in this guide, never benchmarked with the harness |
| **Rapid-MLX** | Native `/v1/messages` | Claims **4.2x Ollama**, 0.08s cached TTFT, ~324 tok/s prefill @8K, ~40 tok/s decode (MTP on by default), **17 tool parsers**, radix-tree + **DeltaNet RNN snapshots** | **The most interesting experiment** — see below |
| vMLX / oMLX | Varies | Prefix caching, paged KV, SSD KV cache aimed at coding agents | Newer, less proven |

**Why Rapid-MLX is the one worth trying.** Its two headline features map startlingly well onto the
exact two failures measured above:

- **"DeltaNet RNN snapshots"** ↔ Qwen3.8-27B is a *hybrid* architecture (`Gated DeltaNet → FFN`
  blocks with periodic gated attention). Hybrid/linear-attention models are exactly the class that
  historically defeats naive KV-cache reuse. A cache designed for DeltaNet is aimed straight at our
  ~82 tok/s prefill and ~5 tok/s decode.
- **"17 tool parsers / 100% tool calling"** ↔ Muse Glimmer returned 1–2 turns with **zero tool
  calls**. If that was Ollama's chat template rather than the model, a purpose-built tool parser is
  the thing that would fix it.

It also lists `qwen3.8-27b-4bit` (~20GB peak RSS) as its 32GB+ tier — i.e. it explicitly targets the
model that timed out here.

> **Treat all of those numbers as vendor claims until this harness reproduces them.** "4.2x faster
> than Ollama" is a project's own README, measured on unstated hardware with an unstated model. The
> whole point of `bench-claude-code.sh` is that engine claims and model leaderboards are exactly the
> things that did not survive contact with a 36GB M3 Pro.

**Recommendation:** keep Ollama as the default — it is the easiest unattended server and its caching
works. Then run one controlled experiment: same model, same tasks, `--base-url` pointed at Rapid-MLX
or vllm-mlx. Because the harness holds the tasks, verifiers, and prompts fixed, swapping only the
engine gives a clean answer to "was that the model's fault or Ollama's?" — which is currently the
single biggest open question in these results.

```bash
# Engine A (incumbent)
./bench-claude-code.sh --models "gemma4:26b-a4b-it-q4_K_M" --tasks "01 02"

# Engine B — same model, same tasks, different server
./bench-claude-code.sh --base-url http://localhost:8000 \
  --models "mlx-community/gemma-4-26B-A4B-it-4bit" --tasks "01 02"
```

### Engine A/B: Ollama vs Rapid-MLX (tested 2026-08-25)

Rapid-MLX advertises "4.2x faster than Ollama, 0.08s cached TTFT, 100% tool calling". Its own
`rapid-mlx recipe` nominates `qwen3.8-27b-4bit` for this exact 36GB Mac at **~41 tok/s** — the very
model Ollama could not finish a task with. Tested head-to-head: same task (`01-fix-bug`), same
verifier, same 900s cap, only the engine swapped (`--base-url`).

| Engine / config | Prefix reuse | Per-turn cost | Result |
|-----------------|--------------|---------------|--------|
| **Ollama 0.32.15** | **21,824 tokens** reused via context checkpoints | ~262s cold, then decode-bound at ~5 tok/s | TIMEOUT |
| Rapid-MLX, recipe defaults | **40 tokens** — `cache_store … stored=False` | 229–246s **every turn**, 0.3–0.8 tok/s | (killed) |
| Rapid-MLX + MTP + 6.5GB cache | 40 tokens — `stored=True`, but lookup still misses | 227s / 254s, 0.3–1.0 tok/s | TIMEOUT, 2 turns |

**Ollama won.** Neither engine finished, but Ollama reused ~21.8k cached tokens per turn while
Rapid-MLX re-prefilled ~21k every turn.

Two distinct defects surfaced, both invisible without reading the engine log:

1. **The prefix cache silently stored nothing.** Rapid-MLX defaults to a **bf16** KV cache, so one
   21k-token entry is ~1.47GB — over its own memory-aware limit of 1208MB (20% of RAM):
   ```
   WARNING: Cache entry too large: 1468.2MB exceeds limit 1208.1MB
   [cache_store] tokens=21226 stored=False
   ```
   Fixable with `--cache-memory-percent 0.35` (note: a **fraction**, not a percentage — passing `35`
   raises `ValueError: max_memory_percent must be in (0, 1]`). Ollama avoids this entirely by
   defaulting to a **q4_0** KV cache, ~6x smaller, so its checkpoints are ~233MB and always fit.

2. **Even once stored, lookup matched only 40 tokens.** Its radix index needs an exact token-prefix
   match, and Claude Code's prompt varies within the first ~40 tokens between turns, so a stored
   21,219-token entry was never reused. Ollama's **position-based context checkpoints tolerate that
   early variation** — which looks like a genuine architectural advantage for agent workloads
   rather than luck.

**MTP could not help.** Enabling it (`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`)
worked — it even injects a GatedDeltaNet chunk-split rollback for this hybrid architecture — but
speculative decoding accelerates generation, and ~97% of each turn was prefill that should not have
been happening. Note also that the defaults **disable spec decode for hybrid models**
(`rapid-mlx info` shows `Spec decode: ✗ disabled (hybrid arch)`, `MTP path: sidecar (opt-in)`), so
the recipe's ~41 tok/s headline is not what the recommended command actually delivers. DFlash is
unavailable at 4-bit (its eligibility check requires ≥8-bit, and an 8-bit build is ~30GB).

**Verdict: keep Ollama.** Its caching is better suited to a 21k-token agent prompt that shifts
slightly every turn. Rapid-MLX's cache design looks tuned for stable-prefix chat. This does not make
it a bad engine — it makes it the wrong engine for *this* workload, which is the sort of thing only
a fixed-task harness can tell you.

### Two things the Claude Code UI implies that are not true locally

Both measured on this setup (2026-08-26, Gemma 4 26B-A4B via Ollama), both contradicting what the
interface suggests.

**1. `--bare` cuts context ~30x, and you almost certainly want it for local models.**

| Config | Input tokens for a trivial prompt |
|--------|-----------------------------------|
| Default (MCP servers + plugins + skills installed) | **20,240** |
| `claude --bare` | **663** |

Capturing the actual HTTP request shows where the weight is: a **6,646-character system prompt and
28 tool definitions**. The *tools*, not the system prompt, are the bulk of it — and every MCP server
you have installed (browser automation, Gmail/Calendar/Drive, mobile, Notion) adds more.

This is not academic. Asked to check email, a 26B model instead announced *"Security Audit of
Codebase"*, called a mobile-automation tool, and finally just recited its own tool list. That is
what a small model does when the tool menu outweighs the request. `--bare` fixes it:

```bash
ANTHROPIC_BASE_URL=http://localhost:11434 ANTHROPIC_AUTH_TOKEN=local \
CLAUDE_CODE_ATTRIBUTION_HEADER=0 \
claude --bare --model gemma4:26b-a4b-it-q4_K_M
```

`--bare` skips hooks, LSP, and plugins but **keeps the file and shell tools working** (verified: it
created a file correctly in 3 turns at 2,218 total input tokens). Skills still resolve explicitly,
so use `/himalaya-email` rather than hoping the model infers the right tool from a 40-item menu.
**Name the tool; don't make a small model choose.** Finer-grained alternatives:
`--disallowedTools "mcp__claude-in-chrome__*"`, or `--mcp-config <minimal.json>`.

> **This likely biased the benchmark above.** Every result in this section ran with the full ~21k
> context. At Qwen3.8-27B's measured ~82 tok/s prefill, a `--bare` run would cut its cold start from
> ~262s to **~27s** — a 10x reduction in the exact cost that timed it out. Glimmer's "1 turn, zero
> tool calls" is also the classic signature of a model swamped by tool definitions. **The dense-model
> verdict above should be treated as provisional until re-run with `--bare`.**

**2. Reasoning effort does not reach a local model.**

Claude Code shows "medium effort" and sends it on every request as a `thinking` field:

```json
"thinking": {"type": "adaptive", "display": "omitted"}
```

But Gemma 4 26B-A4B — which Ollama reports as `["completion","vision","tools","thinking"]`, i.e.
thinking-capable — returned **`thinking_tokens: 0` in all 7 benchmark runs**, including a 17-turn
refactor. Ollama's Anthropic-compat layer does not appear to map Anthropic's `thinking` parameter
onto its own think mode. The effort indicator accurately describes what Claude Code *sent* and
misleads about what *happened*. To get reasoning from a local model, enable it at the Ollama layer
(see [Thinking mode](#thinking-mode)) — the CLI effort control will not do it for you.

### Notes on running it

- **Ollama must be recent enough for the model.** Pulling Qwen 3.8 on Ollama 0.32.1 fails with
  `Error: pull model manifest: 412` — the registry refuses new manifests to old clients. `brew
  upgrade ollama` (0.32.15 worked). The error text says "requires a newer version" but is easy to
  mistake for a network or auth problem.

- **Context length is the #1 failure cause.** Measured on this setup, Claude Code's system prompt
  plus tool definitions costs **~20,200 input tokens before your prompt is even added** (one trivial
  one-turn request billed 20,240 input tokens). Ollama's 4096-token default is therefore hopeless —
  the model silently truncates rather than erroring. The script starts Ollama with
  `OLLAMA_CONTEXT_LENGTH=65536`; 32768 is the practical floor and leaves little room for file reads.
  If Ollama is *already* running the script says so and uses it as-is — kill it first if you want
  the script's settings.
- **That 20k prefill dominates the timings.** Every turn re-prefills it, so the `Eff tok/s` column is
  end-to-end (output tokens ÷ total API time), not raw decode speed. It will read lower than the
  decode figures in the Performance table — that is the point: it is what the agent loop actually
  feels like. Models with high prefill throughput (Ollama measured ~145–244 tok/s prefill) do
  disproportionately well here.
- **Ignore `total_cost_usd` in the JSON** — Claude Code prices local tokens as if they were API
  tokens. It is useful only as "what this run would have cost on the API" (the trivial test above:
  $0.11). Local runs cost nothing but electricity.
- Claude Code prints `[claude-code:unrecognized_model]` to stderr for any non-Anthropic model name.
  It is a harmless warning, not a failure — runs complete normally.
- Runs use `--dangerously-skip-permissions` so they don't block on prompts. Every run happens inside
  a throwaway fixture directory under `bench/results/`, never in your real repo.
- Models are unloaded between candidates (`keep_alive: 0`) so two 18GB models never share 36GB.
- `--timeout` (default 900s) caps each run; a killed run is recorded as `TIMEOUT` rather than
  silently hanging the matrix.

## Troubleshooting

### OOM / CUDA out of memory
Reduce context: `-c 32768` or `-c 16384`. The KV cache scales with context size.

### Model loads, then dies on the first request (12GB, `free = 0`)
On a 12GB card the Qwen3.6-35B-A3B IQ2 pick fits its *weights* but leaves almost no VRAM free (`llama_memory_breakdown` shows `free = 0`). It loads and starts listening, then OOM-exits the moment a real request needs more — Claude Code's ~15K-token system prompt, a longer context, or a CUDA graph capture. Two symptoms: (1) restarting the server OOMs because the previous instance's VRAM hasn't been released yet (WSL2 is slow to free it — wait ~5s and confirm `nvidia-smi` shows idle before restarting); (2) it works for one tiny prompt then crashes on a bigger one.

**Fix: offload some MoE experts to CPU for headroom** with `--n-cpu-moe` (`-ncmoe`). Measured on the 4070 Ti at 128K context: `-ncmoe 16` → ~8.8GB, ~40 tok/s; `-ncmoe 8` → ~10.6GB, **~77 tok/s** (fewer CPU experts = faster *and* more VRAM used). This is exactly what `start-claude-windows.sh` does. Raise `CTX` for more agent room and raise `-ncmoe` to pay for it in VRAM. (Cheap for an MoE — only ~3B params are active per token, so CPU-side expert compute is modest.)

### `400: 'type' of tool must be 'function'` (Claude Code + local model)
Claude Code offered the model an Anthropic **server-side** tool — **`WebSearch`** or **`WebFetch`** (their tool `type` is `web_search_*`/`web_fetch_*`, not `function`) — and llama-server rejected it. These tools run on Anthropic's servers and can't work against a local model anyway. **Fix:** add `--disallowedTools WebSearch WebFetch` to your `claude` launch (`start-claude-windows.sh` already does).

### LiteLLM won't start after an OS upgrade (`ModuleNotFoundError: No module named 'litellm'`)
A distro upgrade that bumps the system Python (e.g. Ubuntu 26.04 → Python 3.14) orphans a pip-installed `litellm` from the old version. Reinstall into a venv on the new Python:
```bash
python3 -m venv ~/litellm-venv
~/litellm-venv/bin/pip install 'litellm[proxy]'   # bootstrap pip via get-pip.py first if venv ensurepip is missing
```
Then run `~/litellm-venv/bin/litellm --config …`. `start-claude-windows.sh` auto-detects `~/litellm-venv/bin/litellm`. A **driver-only** update needs no rebuild of llama.cpp; a CUDA-toolkit **major** bump might.

### Slow on Apple Silicon (< 10 tok/s)
Check `memory_pressure` — if free memory is below 20%, you're swap-thrashing. Close memory-hungry apps (Chrome is usually the biggest offender) or switch to the 9B model.

### Slow prompt processing
Make sure all layers are on GPU (`offloaded N/N layers to GPU` in logs). If not, reduce context or use a smaller quant.

### Gemma 4 outputs `<unused25>` garbage in llama.cpp
Known bug ([#21321](https://github.com/ggml-org/llama.cpp/issues/21321)). Gemma 4 gets stuck generating thinking tokens that llama.cpp doesn't filter correctly. **Workaround: use Ollama instead of llama.cpp for Gemma 4 on Mac.** Ollama's MLX backend handles the thinking tokens correctly. Qwen3.5 models are unaffected.

### Ollama CUDA crashes with MoE models
~~Known bug ([#14444](https://github.com/ollama/ollama/issues/14444)).~~ **Fixed in Ollama v0.17.5.** Ollama now works with MoE models (Qwen3.5 and Gemma 4) on NVIDIA GPUs. Update to v0.17.5+ if you hit this.

### Cursor SSRF blocked
Cursor routes requests through their servers and blocks localhost/private IPs. Use cloudflared or ngrok to create a public HTTPS tunnel.

### Cursor "Model name is not valid"
Cursor has client-side model name validation. Use LiteLLM proxy to map an accepted name (like `deepseek-r1-0528`) to your local model.

### Cursor Agent mode doesn't use custom model
This is by design — Agent mode only works through Cursor's backend. Use Chat/Cmd+K mode, or switch to Claude Code or Cline for agentic workflows with local models.

### WSL only sees half your RAM
WSL2 defaults to 50% of system RAM. Create `C:\Users\<you>\.wslconfig`:
```ini
[wsl2]
memory=40GB
swap=8GB
```
Then restart WSL: `wsl --shutdown`

## Multi-Token Prediction (MTP) — 1.5–3× speedup

MTP speculative decoding uses a small "drafter" head to propose multiple tokens at once, which the main model verifies in a single forward pass. Outputs are identical to non-MTP generation — it's pure throughput, no quality loss. Reported speedups: ~2× for Gemma 4 31B coding tasks, 1.5–2.9× for Qwen 3.x models.

### Ollama (Gemma 4 — easiest path)

Ollama added Gemma 4 MTP support via [PR #15980](https://github.com/ollama/ollama/pull/15980). Pre-built MTP models are on the Ollama library:

```bash
ollama pull gemma4:31b-coding-mtp-bf16
ollama run gemma4:31b-coding-mtp-bf16
```

Google ships official drafters and Ollama wires them up automatically. **Caveat (measured 2026-07-24):** Ollama publishes this only as **bf16 (~62GB)** — there is no quantized MTP tag, so it **does not fit a 36GB Mac** (the dense Q4 31B already swap-thrashes at ~6 tok/s there). MTP-on-Ollama for Gemma 4 is realistically a ≥64GB-machine feature; on 36GB, use the 26B-A4B MoE (no MTP needed, ~33 tok/s) or the llama.cpp Q4 MTP path below.

### llama.cpp (beta — Qwen 3.x and Gemma 4)

llama.cpp MTP support landed in [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673), currently beta. You need a build that includes the PR, plus an MTP-bundled GGUF:

```bash
# Download a community MTP-bundled GGUF (drafter weights baked in)
huggingface-cli download havenoammo/Qwen3.6-35B-A3B-MTP-GGUF \
  Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf --local-dir ./models

# Serve with MTP enabled
./llama-server \
  -m models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 -c 131072 -fa on \
  --parallel 1 \
  --spec-type mtp --spec-draft-n-max 2 \
  --cache-type-k q4_0 --cache-type-v q4_0
```

**Caveats:**
- `--parallel 1` is required — MTP doesn't support multi-slot serving yet.
- **The speedup lands on *dense* models, not on the A3B MoE that fits 12GB.** Measured on Qwen 3.6 via PR #22673: **~1.73× on the 27B dense, only ~1.17× on the 35B-A3B MoE**; an independent RTX 3090 run found *no net speedup* on Ampere + A3B ([thc1006/qwen3.6-speculative-decoding-rtx3090](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)). So for the 12GB IQ2 MoE pick, don't expect much from plain MTP — dense Gemma 4 31B / Qwen 3.6 27B are where it pays off.
- **Better MoE path — NextN speculative decoding (fork).** The [AtomicBot-ai TurboQuant fork](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) implements Qwen 3.6 **NextN** auxiliary-head drafting reporting **+28–36% on the 35B-A3B MoE** (and Gemma 4 MTP at +30–50%), with CUDA kernels — meaningfully better than upstream MTP's 1.17× on that model. Not merged upstream; see [KV cache compression](#kv-cache-compression-turboquant--now-on-cuda-via-forks).
- MTP-bundled GGUFs are community-built; quality may vary.

## KV cache compression (TurboQuant) — now on CUDA via forks

Google's [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) compresses the KV cache to 3 bits with zero accuracy loss. This matters for local inference because token generation is **memory bandwidth bound** — every token reads the entire KV cache. Smaller cache = less data to read = faster tok/s.

We currently use `--cache-type-k q4_0 --cache-type-v q4_0` (4-bit KV cache). Moving to a 3-bit TurboQuant cache would give us:

- **Faster generation** — ~25% less data to read per token during attention
- **Higher quality quants** — freed VRAM means you could use Q5_K_M or Q6_K instead of Q4_K_M for model weights, getting better output at the same VRAM budget
- **Longer context** — the 9B model could feasibly run 262K context on 12GB VRAM instead of being limited to 131K

| Scenario | Current (q4_0 cache) | With TurboQuant (3-bit cache) |
|---|---|---|
| Qwen 9B + 131K on 12GB | Q4_K_M, ~65 tok/s | Could use Q6_K, faster attention |
| Qwen 9B + 262K on 12GB | OOM | Feasible |
| Nemotron 4B + 262K on 8GB | Tight | Comfortable |

**Status (updated 2026-07):**
- **Merged upstream:** [PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038) — Hadamard rotation before KV caching (the core TurboQuant idea). Works on all backends including Metal and CUDA. Makes existing `q4_0` cache more accurate at the same memory footprint — a free quality upgrade when you update llama.cpp.
- **Actual 3-bit KV types — CUDA now works, but via forks, not upstream yet.** [PR #21089](https://github.com/ggml-org/llama.cpp/pull/21089) (TBQ3_0/TBQ4_0) was CPU-only in review; since then several forks have shipped **working CUDA kernels**:
  - [AtomicBot-ai/atomic-llama-cpp-turboquant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) — TURBO2/3/4_0 KV + TQ4_1S weight quant with CUDA kernels (~4.3× KV compression), **plus Qwen 3.6 NextN and Gemma 4 MTP** in the same build.
  - [AmesianX/TurboQuant](https://github.com/AmesianX/TurboQuant) — reports ~5.2× KV memory reduction, near-lossless.
  - [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) — TurboQuant with CUDA support.
  - Real-world: a turbo4/q8_0 hybrid KV cache hit **~40.6 tok/s on an RTX 3070** with full 262K context ([writeup](https://blog.lucad.cloud/articles/turboquant-mtp-get-406-toks-out-of-qwen36)).
- **Caveat:** Symmetric TurboQuant degrades quality on Qwen models — K compression (not V) drives the loss (Qwen has a 10–60× K/V magnitude ratio). Use asymmetric configs (e.g. `q8_0` for K, `turbo3`/`turbo4` for V). The [turboquant_plus](https://github.com/TheTom/turboquant_plus) fork measured turbo4 at +0.23% PPL and turbo3 at +1.06% vs a q8_0 baseline.
- **For our 12GB card:** this is now worth trying — an asymmetric turbo cache frees enough VRAM to push the Qwen3.6-IQ2 pick past its ~57K context ceiling, or to run the weights at a higher quant. It's still fork territory (build required), so treat it as an experiment, not the default `benchmark.sh` config, until it lands upstream.

## Tangential: Hybrid Local + Cloud Agents

*This guide is about running models **locally**. This section is secondary — it's for the case where you mix in a cloud model (e.g. Claude) for the hard ~20% of tasks and want to keep that cheap and observable. Skip it if you're fully local.*

**Decouple the brain from the hands.** An agent is two parts: the reasoning loop ("brain") and the code execution / tool calls ("hands"). They don't have to live in the same place. Run the brain locally (the models in this guide) for routine steps — grep, read, edit, run tests — and escalate to a cloud model only when a step is genuinely hard. The hands (a sandbox where commands run) can stay local (Docker/gVisor) or be managed, e.g. [Cloudflare Sandboxes](https://blog.cloudflare.com/claude-managed-agents/). Most agent *steps* are cheap; paying frontier prices for `ls` and `sed` is the main waste. Keep those local, spend cloud tokens only on the hard reasoning step plus a final verify.

**If you do call a cloud API, put a gateway in front of it** — for cost ceilings and visibility:

- **[Cloudflare AI Gateway](https://developers.cloudflare.com/ai-gateway/)** — a config-defined proxy giving per-request token/cost logging, fleet-wide spend caps, rate limiting, caching, and provider routing/fallbacks. The better choice for a standing control plane, especially if your sandbox already lives in Cloudflare. Caveat: its caching is whole-response (identical request → cached response), which rarely hits in an agent loop where the history grows every turn — so treat it as your governance + observability layer, not your prompt-cache layer.
- **[alxsuv/pino](https://github.com/alxsuv/pino)** — a tiny (~500 LOC, zero-dep) local proxy between Claude Code and the Anthropic API. It logs request/response bodies and injects prompt-cache breakpoints on content the client ships uncached (Claude Code re-sends its ~5–15k-token tool catalog every turn), converting repeated static content from 1× to 0.1× cache-read cost. It exists to rewrite a client you *can't* modify — if you're writing your own agent loop, place the cache breakpoints in your own request code and skip the proxy.

**Prompt caching is the big lever on cloud cost.** Anthropic cache reads cost ~0.1× of base input; writes cost 1.25× (5-min TTL) or 2× (1-hour). An agent loop resends the same system prompt + tool definitions every step, so caching that stable prefix is ~90% off on that chunk. Every response's `usage` object reports `cache_read_input_tokens` / `cache_creation_input_tokens` / `input_tokens` / `output_tokens` — if `cache_read` is 0 across steps, a prefix change is silently breaking the cache.

**Why this matters for a local guide:** local-first keeps the marginal cost of "let it run" near zero, which is exactly what makes longer-running, less-supervised agents affordable. Cloud is the escape hatch for the hard step — gated behind a budget and a gateway.

## Credits

- [@sudoingX](https://x.com/sudoingX) for the optimized llama-server flags and Qwen3.5 benchmarks
- [llama.cpp](https://github.com/ggml-org/llama.cpp) by ggml-org
- [Qwen3.5](https://github.com/QwenLM/Qwen3.5) by Alibaba Qwen team
- [Gemma 4](https://ai.google.dev/gemma) by Google DeepMind
- [Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) by NVIDIA
- [unsloth](https://huggingface.co/unsloth) for GGUF quantizations
- [mlx-community](https://huggingface.co/mlx-community) for MLX quantizations
- [vllm-mlx](https://github.com/waybarrios/vllm-mlx) for MLX inference with Anthropic API
- [Ollama](https://ollama.com/) for easy local model management
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) + [TabbyAPI](https://github.com/theroyallab/tabbyAPI) for fast CUDA inference
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) by NVIDIA for optimized engine compilation
- [LiteLLM](https://github.com/BerriAI/litellm) for the proxy workaround

## License

MIT
