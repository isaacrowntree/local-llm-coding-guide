# Local LLM Coding Guide

Run local LLMs as a coding assistant on consumer hardware. Supports Qwen3.5 and Gemma 4 via llama.cpp, Ollama (MLX), or vllm-mlx.

Tested on:
- **Windows/WSL2:** RTX 4070 Ti (12GB), Intel Core Ultra 9 285K, 48GB DDR5
- **macOS:** M3 MacBook Pro, 36GB unified memory

## Performance

| GPU | Model | Tok/s | Context | Memory Used |
|-----|-------|-------|---------|-------------|
| RTX 4070 Ti 12GB | Nemotron 3 Nano 4B Q4_K_M | TBD | 262K | ~5GB |
| RTX 4070 Ti 12GB | Qwen3.5-9B Q4_K_M | ~65 tok/s | 131K | 7.8GB |
| RTX 3060 12GB | Qwen3.5-9B Q4_K_M | ~43 tok/s | 128K | ~7.8GB |
| RTX 3090 24GB | Qwen3.5-27B Q4_K_M | ~30 tok/s | 262K | ~18GB |
| M3 Pro 36GB | **Qwen3.5-35B-A3B Q4_K_M** | **~29 tok/s** | 131K | **~22GB** |
| M3 Pro 36GB | Qwen3.5-9B Q4_K_M | ~20 tok/s | 131K | ~7GB |
| M3 Pro 36GB | Qwen3.5-27B Q4_K_M | ~9 tok/s* | 131K | ~18GB |
| M3 Pro 36GB | **Gemma 4 26B-A4B Q4_K_M (Ollama MLX)** | **~31 tok/s** | 256K | **~17GB** |
| M3 Pro 36GB | Gemma 4 31B Q4_K_M | TBD | 256K | ~18GB |

*The dense 27B model is slower than the 35B-A3B on 36GB machines due to higher memory bandwidth requirements. The 35B-A3B (MoE) is faster *and* smarter — see [Why MoE?](#why-moe-mixture-of-experts) below.

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

**For macOS with 32GB+ RAM (recommended):**
```bash
pip install huggingface-hub
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-Q4_K_M.gguf --local-dir ./models
```

**For NVIDIA GPUs (8-12GB VRAM):**
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
  -m models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
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
  -m models/Qwen3.5-9B-Q4_K_M.gguf \
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
| Gemma 4 E4B | 8B | 4.5B (dense) | ~5GB | 128K | TBD |
| Qwen3.5-9B | 9B | 9B (dense) | 5.3GB | 131K | GPT-4o-mini / Haiku |
| Qwen3.5-27B | 27B | 27B (dense) | 16GB | 131K | Sonnet-ish |
| **Gemma 4 26B-A4B** | **26B** | **4B (MoE)** | **16.9GB** | **256K** | **TBD** |
| Gemma 4 31B | 31B | 31B (dense) | 18.3GB | 256K | TBD |
| **Qwen3.5-35B-A3B** | **35B** | **3B (MoE)** | **22GB** | **131K** | **Sonnet 4.5** |

The 35B-A3B is the sweet spot for Apple Silicon: it's **faster than the 9B** (29 vs 20 tok/s on M3 Pro) because it only computes 3B params per token, yet **smarter than the 27B** because it draws from 35B total parameters. It [beats Sonnet 4.5 on several benchmarks](https://venturebeat.com/technology/alibabas-new-open-source-qwen3-5-medium-models-offer-sonnet-4-5-performance) including instruction following and visual reasoning.

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
| **ExLlamaV3 + TabbyAPI** | **~100-130** | EXL3 (CUDA-only) | Not yet | `git clone` + `start.sh` |
| **TensorRT-LLM** | **~80-95** | TRT engine / HF | Not yet | Docker container |
| **llama.cpp** | ~65 | GGUF (universal) | Yes (day-0) | Build or download binary |
| Ollama | ~62 | GGUF (llama.cpp backend) | Yes | `brew install ollama` |

**Why two formats?** GGUF is universal (CUDA, Metal, CPU, Vulkan) using generic compute kernels. EXL3 trades portability for speed — it uses hand-tuned CUDA kernels optimized for NVIDIA memory hierarchy, so it only runs on NVIDIA GPUs but is significantly faster.

### ExLlamaV3 + TabbyAPI (fastest for NVIDIA)

ExLlamaV3 uses the EXL3 format with calibration-based quantization — it measures which layers matter most and allocates more bits to them. TabbyAPI serves it with an OpenAI-compatible API.

> **Note:** ExLlamaV2 is archived. ExLlamaV3 + EXL3 is the active project. Gemma 4 is not yet supported — use llama.cpp for Gemma 4 on CUDA.

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

**Claude Code integration:**
```bash
ANTHROPIC_BASE_URL=http://localhost:5000/v1 \
ANTHROPIC_AUTH_TOKEN=local \
claude --model openai/qwen
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
| Gemma 4 * | — | — | — | Not yet supported |

\*ExLlamaV3 supports Gemma 2 and 3, but not Gemma 4 yet. Watch the [ExLlamaV3 repo](https://github.com/turboderp-org/exllamav3) for updates.

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

**Claude Code integration:**
```bash
ANTHROPIC_BASE_URL=http://localhost:8000/v1 \
ANTHROPIC_AUTH_TOKEN=local \
claude --model openai/qwen
```

**Performance vs other engines (estimated, 8-9B INT4, RTX 4070 Ti):**

| Engine | ~Tok/s | Notes |
|--------|--------|-------|
| ExLlamaV3 (EXL3) | ~100-130 | Custom CUDA kernels, single-user champion |
| TensorRT-LLM (TRT engine) | ~80-95 | GPU-specific compiled graph, higher VRAM overhead |
| TensorRT-LLM (PyTorch) | ~75-85 | No build step, good middle ground |
| llama.cpp (GGUF) | ~65 | Universal format, broadest compatibility |

TRT-LLM's advantage grows with batched/concurrent inference. For single-user coding, ExLlamaV3 may match or exceed it due to its hand-tuned decode kernels. TRT-LLM becomes the clear winner when serving multiple users or running longer-context workloads where its compiled attention kernels shine.

## Inference Engines (macOS Apple Silicon)

**Ollama (MLX) is recommended for macOS.** llama.cpp has an [open bug](https://github.com/ggml-org/llama.cpp/issues/21321) where Gemma 4 outputs only thinking tokens (`<unused25>`) — use Ollama until this is fixed.

| Engine | Decode Speed | Prefill Speed | Format | Claude Code API | Setup |
|--------|-------------|---------------|--------|-----------------|-------|
| **Ollama 0.20+** | **~31 tok/s (tested)** | **~99 tok/s** | Auto (MLX) | OpenAI | `brew install ollama` |
| llama.cpp | ~29 tok/s (Qwen only) | ~70 tok/s | GGUF | OpenAI (needs `openai/` prefix) | `brew install llama.cpp` |
| vllm-mlx | MLX-native | Good | MLX | **Native Anthropic** (no proxy) | `pip install` from git |

> **Note:** llama.cpp works fine with Qwen3.5 models on Mac. The thinking token bug only affects Gemma 4.

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
ANTHROPIC_BASE_URL=http://localhost:11434/v1 \
ANTHROPIC_AUTH_TOKEN=local \
claude --model openai/gemma4:26b-a4b-it-q4_K_M
```

### vllm-mlx (recommended for Claude Code)

vllm-mlx exposes a native Anthropic `/v1/messages` endpoint — no LiteLLM proxy or `openai/` prefix needed.

```bash
# Install
pip install git+https://github.com/AnyLLM/vllm-mlx.git

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
| `start-claude-local.sh` | 1 | Any | Launch Claude Code with local model (auto-detects) |
| `start-remote.sh` | 2 | Linux/WSL | Tunnel llama-server for remote access |
| `start-cursor-local.sh` | 3 | Linux/WSL | LiteLLM proxy + tunnel for Cursor |
| `stop-all.sh` | — | Any | Kill everything |

Always run `start-server.sh` or `start-server-mac.sh` first, then pick your flow.

## IDE Integration

### Claude Code (recommended for agentic coding)

**Flow 1: Local (same machine)**

```bash
# Terminal 1: start the server
./start-server-mac.sh   # macOS
./start-server.sh       # Linux/WSL

# Terminal 2: Claude Code with local Qwen
./start-claude-local.sh
```

Or manually:
```bash
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude --model openai/qwen
```

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
claude --model openai/qwen
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

## Model Selection Guide

| Memory | Recommended Model | Type | Q4 Size | Tok/s (approx) | Quality Tier |
|--------|-------------------|------|---------|-----------------|-------------|
| 8GB VRAM | Qwen3.5-9B | Dense | 5.3GB | ~43-65 | Haiku |
| 8GB VRAM | Nemotron 3 Nano 4B (alt) | Hybrid Mamba-2 | ~2.5GB | faster* | Below Haiku |
| 12GB VRAM | Qwen3.5-9B | Dense | 5.3GB | ~43-65 | Haiku |
| 16GB VRAM | Qwen3.5-9B | Dense | 5.3GB | ~43-65 | Haiku |
| 24GB VRAM | Qwen3.5-27B | Dense | 16GB | ~30 | Sonnet-ish |
| 24GB VRAM | Gemma 4 26B-A4B | MoE | 16.9GB | TBD | TBD |
| 32GB+ (Apple Silicon) | **Qwen3.5-35B-A3B** | **MoE** | **22GB** | **~29 (llama.cpp)** | **Sonnet 4.5** |
| 32GB+ (Apple Silicon) | **Gemma 4 26B-A4B** | **MoE** | **17GB** | **~31 (Ollama MLX)** | **Sonnet 4.5** |
| 36GB+ (Apple Silicon) | Gemma 4 31B | Dense | 20GB | TBD | TBD |

\*Nemotron 3 Nano 4B uses a hybrid Mamba-2 + Transformer architecture (mostly Mamba-2 layers with just 4 attention layers). It's significantly faster than the 9B, uses only ~5GB VRAM, and supports 262K context. The tradeoff is lower coding quality — it's designed for edge agents and local assistants rather than deep reasoning. Use it when you want maximum speed or need the longer context window. Run it with `./start-server.sh nemotron`.

### Why not the dense 27B on Apple Silicon?

On a 36GB M3 Pro, the dense 27B model uses ~18GB for weights + 2.3GB for KV cache, leaving very little headroom. In practice this causes **swap thrashing** (~1.7 tok/s) when other apps are running. The 35B-A3B MoE is larger on disk (22GB) but faster at inference because it only activates 3B parameters per token.

### Honest comparison to premium models

The **Qwen3.5-35B-A3B** sits roughly in the **Sonnet 4.5 tier** — it beats Sonnet 4.5 on instruction following (IFBench) and is competitive on coding benchmarks. It handles single-file tasks, refactoring, bug fixes, and test writing well.

The **Qwen3.5-9B** sits in the **GPT-4o-mini / Haiku tier**. Good for fast completions, quick edits, boilerplate, and explanations.

Both still fall short of Claude Opus or GPT-4.5 on complex multi-step reasoning and autonomous agentic workflows. Best strategy: use local models for quick edits and premium APIs for hard problems.

## Troubleshooting

### OOM / CUDA out of memory
Reduce context: `-c 32768` or `-c 16384`. The KV cache scales with context size.

### Slow on Apple Silicon (< 10 tok/s)
Check `memory_pressure` — if free memory is below 20%, you're swap-thrashing. Close memory-hungry apps (Chrome is usually the biggest offender) or switch to the 9B model.

### Slow prompt processing
Make sure all layers are on GPU (`offloaded N/N layers to GPU` in logs). If not, reduce context or use a smaller quant.

### Gemma 4 outputs `<unused25>` garbage in llama.cpp
Known bug ([#21321](https://github.com/ggml-org/llama.cpp/issues/21321)). Gemma 4 gets stuck generating thinking tokens that llama.cpp doesn't filter correctly. **Workaround: use Ollama instead of llama.cpp for Gemma 4 on Mac.** Ollama's MLX backend handles the thinking tokens correctly. Qwen3.5 models are unaffected.

### Ollama CUDA crashes with MoE models
Known bug ([#14444](https://github.com/ollama/ollama/issues/14444)). Use llama.cpp directly instead of ollama for Qwen3.5 MoE models.

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

## What's Next: TurboQuant (KV Cache Compression)

Google's [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) compresses the KV cache to 3 bits with zero accuracy loss. This matters for local inference because token generation is **memory bandwidth bound** — every token reads the entire KV cache. Smaller cache = less data to read = faster tok/s.

We currently use `--cache-type-k q4_0 --cache-type-v q4_0` (4-bit KV cache). Once TurboQuant lands in llama.cpp, switching to 3-bit cache would give us:

- **Faster generation** — ~25% less data to read per token during attention
- **Higher quality quants** — freed VRAM means you could use Q5_K_M or Q6_K instead of Q4_K_M for model weights, getting better output at the same VRAM budget
- **Longer context** — the 9B model could feasibly run 262K context on 12GB VRAM instead of being limited to 131K

| Scenario | Current (q4_0 cache) | With TurboQuant (3-bit cache) |
|---|---|---|
| Qwen 9B + 131K on 12GB | Q4_K_M, ~65 tok/s | Could use Q6_K, faster attention |
| Qwen 9B + 262K on 12GB | OOM | Feasible |
| Nemotron 4B + 262K on 8GB | Tight | Comfortable |

**Status (April 2026):**
- **Merged:** [PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038) — Hadamard rotation before KV caching (the core TurboQuant idea). Works on all backends including Metal and CUDA. Makes existing `q4_0` cache more accurate at the same memory footprint — a free quality upgrade when you update llama.cpp.
- **In review:** [PR #21089](https://github.com/ggml-org/llama.cpp/pull/21089) — Actual 3-bit KV cache types (TBQ3_0/TBQ4_0). CPU-only so far — CUDA support is being developed, Metal support not yet started. When it merges with GPU support, it's a free upgrade — just change the `--cache-type-k` and `--cache-type-v` flags.
- **Caveat:** Symmetric TurboQuant degrades quality on Qwen models. Asymmetric configs (q8_0 for K, turbo3 for V) are recommended when TBQ types land.

## Credits

- [@sudoingX](https://x.com/sudoingX) for the optimized llama-server flags and Qwen3.5 benchmarks
- [llama.cpp](https://github.com/ggml-org/llama.cpp) by ggml-org
- [Qwen3.5](https://github.com/QwenLM/Qwen3.5) by Alibaba Qwen team
- [Gemma 4](https://ai.google.dev/gemma) by Google DeepMind
- [Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) by NVIDIA
- [unsloth](https://huggingface.co/unsloth) for GGUF quantizations
- [mlx-community](https://huggingface.co/mlx-community) for MLX quantizations
- [vllm-mlx](https://github.com/AnyLLM/vllm-mlx) for MLX inference with Anthropic API
- [Ollama](https://ollama.com/) for easy local model management
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) + [TabbyAPI](https://github.com/theroyallab/tabbyAPI) for fast CUDA inference
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) by NVIDIA for optimized engine compilation
- [LiteLLM](https://github.com/BerriAI/litellm) for the proxy workaround

## License

MIT
