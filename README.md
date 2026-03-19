# Local LLM Coding Guide

Run Qwen3.5 locally as a coding assistant on consumer hardware.

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

| Model | Total Params | Active Params | Q4_K_M Size | Quality Tier |
|-------|-------------|---------------|-------------|-------------|
| Nemotron 3 Nano 4B | 4B | 4B (hybrid Mamba-2) | ~2.5GB | Below Haiku (edge/agent) |
| Qwen3.5-9B | 9B | 9B (dense) | 5.3GB | GPT-4o-mini / Haiku |
| Qwen3.5-27B | 27B | 27B (dense) | 16GB | Sonnet-ish |
| **Qwen3.5-35B-A3B** | **35B** | **3B** | **22GB** | **Sonnet 4.5** |

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

## Supported Flows

```
Flow 1: Claude Code (local)
  Claude Code -> localhost:8080 -> llama-server (Qwen3.5)
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
| `start-server.sh` | All | Linux/WSL | Start llama-server (default: Qwen3.5-9B, or `./start-server.sh nemotron`) |
| `start-server-mac.sh` | All | macOS | Start llama-server (auto-picks 35B-A3B or 9B based on RAM) |
| `start-claude-local.sh` | 1 | Any | Launch Claude Code with local Qwen (auto-detects model) |
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

| Memory | Recommended Model | Type | Q4_K_M Size | Tok/s (approx) | Quality Tier |
|--------|-------------------|------|-------------|-----------------|-------------|
| 8GB VRAM | Qwen3.5-9B | Dense | 5.3GB | ~43-65 | Haiku |
| 8GB VRAM | Nemotron 3 Nano 4B (alt) | Hybrid Mamba-2 | ~2.5GB | faster* | Below Haiku |
| 12GB VRAM | Qwen3.5-9B | Dense | 5.3GB | ~43-65 | Haiku |
| 16GB VRAM | Qwen3.5-9B | Dense | 5.3GB | ~43-65 | Haiku |
| 24GB VRAM | Qwen3.5-27B | Dense | 16GB | ~30 | Sonnet-ish |
| 32GB+ (Apple Silicon) | **Qwen3.5-35B-A3B** | **MoE** | **22GB** | **~29** | **Sonnet 4.5** |
| 36GB+ (Apple Silicon) | **Qwen3.5-35B-A3B** | **MoE** | **22GB** | **~29** | **Sonnet 4.5** |

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

## Credits

- [@sudoingX](https://x.com/sudoingX) for the optimized llama-server flags and Qwen3.5 benchmarks
- [llama.cpp](https://github.com/ggml-org/llama.cpp) by ggml-org
- [Qwen3.5](https://github.com/QwenLM/Qwen3.5) by Alibaba Qwen team
- [Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) by NVIDIA
- [unsloth](https://huggingface.co/unsloth) for GGUF quantizations
- [LiteLLM](https://github.com/BerriAI/litellm) for the proxy workaround

## License

MIT
