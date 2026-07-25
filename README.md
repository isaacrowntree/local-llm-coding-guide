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

*The dense 27B model is slower than the 35B-A3B on 36GB machines due to higher memory bandwidth requirements. The 35B-A3B (MoE) is faster *and* smarter — see [Why MoE?](#why-moe-mixture-of-experts) below.

† Measured 2026-07-24 on M3 Pro 36GB. Raw `mlx-lm` decodes Qwen 3.6 35B-A3B at **~48 tok/s** vs Ollama MLX's **~35 tok/s** (both matched over a ~4000-token generation) — Ollama trades ~30% decode speed for its `q4_0` KV-cache + serving convenience. Ollama's prefill is much higher (~145 vs an unreliable CLI figure for mlx-lm). Pick Ollama for ease, `mlx-lm` if you want the fastest single-stream decode.

‡ The dense 31B **swap-thrashes on 36GB** (~18GB weights leaves too little headroom) — measured ~6 tok/s, i.e. 5× slower than the 26B-A4B MoE. Not recommended on 36GB; use the 26B-A4B MoE instead.

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
| llama.cpp | ~29 tok/s (Qwen) | ~70 tok/s | GGUF | OpenAI — needs LiteLLM proxy | `brew install llama.cpp` |
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
| `start-remote.sh` | 2 | Linux/WSL | Tunnel llama-server for remote access |
| `start-cursor-local.sh` | 3 | Linux/WSL | LiteLLM proxy + tunnel for Cursor |
| `stop-all.sh` | — | Any | Kill everything |

Always run `start-server.sh` or `start-server-mac.sh` first, then pick your flow.

## IDE Integration

### Claude Code (recommended for agentic coding)

> **How Claude Code talks to a local model:** Claude Code speaks **only the Anthropic Messages API** (`/v1/messages`) — it is *not* an OpenAI client. That splits local servers into two cases:
> - **Native Anthropic — connect directly, no proxy, no prefix.** **Ollama 0.14+** and **vllm-mlx** expose `/v1/messages`. Point Claude Code straight at them with the plain model name (note: base URL has **no `/v1`**):
>   `ANTHROPIC_BASE_URL=http://localhost:11434 ANTHROPIC_AUTH_TOKEN=local claude --model qwen3.6:35b`
> - **OpenAI-only — needs a LiteLLM proxy.** **llama.cpp (`llama-server`), TabbyAPI, TensorRT-LLM** expose only OpenAI `/v1/chat/completions`, which Claude Code cannot consume directly. Run a [LiteLLM proxy](#cursor-limited--agent-mode-unsupported) to translate Anthropic↔OpenAI, then point Claude Code at the proxy.
>
> The `openai/<model>` prefix is a **LiteLLM routing convention** (it tells LiteLLM the backend is OpenAI-format) — it belongs in the LiteLLM config, never in a `claude --model` that points straight at Ollama or vllm-mlx. On a Mac, **Ollama is the zero-proxy path** and is what the rest of this section assumes.

#### Windows / WSL2 + NVIDIA (validated 2026-07-25)

llama.cpp is OpenAI-only, so the Windows path is: **`llama-server` → LiteLLM (Anthropic bridge) → Claude Code**. One script sets up all three:

```bash
./start-claude-windows.sh --dangerously-skip-permissions
# or headless:  ./start-claude-windows.sh -p "fix the failing test in test_foo.py"
```

Or copy-paste the launch string yourself (once `llama-server` + LiteLLM are up on :8080 / :4000):

```bash
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_AUTH_TOKEN=local ANTHROPIC_API_KEY=local \
claude --model qwen3.6-local --dangerously-skip-permissions \
  --disallowedTools WebSearch WebFetch
```

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
