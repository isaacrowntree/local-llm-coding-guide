# Local LLM Coding Guide

Run Qwen3.5-9B locally as a coding assistant on consumer GPUs (8-12GB VRAM).

Tested on: RTX 4070 Ti (12GB), Intel Core Ultra 9 285K, 48GB DDR5, WSL2 on Windows 11.

## Performance

| GPU | Model | Tok/s | Context | VRAM Used |
|-----|-------|-------|---------|-----------|
| RTX 4070 Ti 12GB | Qwen3.5-9B Q4_K_M | ~65 tok/s | 131K | 7.8GB |
| RTX 3060 12GB | Qwen3.5-9B Q4_K_M | ~43 tok/s | 128K | ~7.8GB |
| RTX 3090 24GB | Qwen3.5-27B Q4_K_M | ~30 tok/s | 262K | ~18GB |

## Quick Start

### 1. Install llama.cpp

**Option A: Pre-built binary (recommended)**

Download the latest release for your platform from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases).

**Option B: Build from source with CUDA**

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Find your GPU's compute capability: https://developer.nvidia.com/cuda-gpus
# RTX 4070 Ti = 89, RTX 3090 = 86, RTX 3060 = 86
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89"
cmake --build build -j$(nproc)
```

### 2. Download the model

```bash
pip install huggingface-hub
huggingface-cli download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir ./models
```

The Q4_K_M quant is 5.3GB — fits on any 8GB+ GPU.

### 3. Start the server

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
  --reasoning-budget 0 \
  --metrics
```

**Flag breakdown:**

| Flag | Purpose |
|------|---------|
| `-ngl 99` | Offload all layers to GPU |
| `-c 131072` | Context window (model supports up to 262K) |
| `--cache-type-k q4_0 --cache-type-v q4_0` | Quantize KV cache to fit large context in VRAM |
| `--reasoning-budget 0` | Disable thinking mode for faster responses |
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

## IDE Integration

### Claude Code (recommended for agentic coding)

Works out of the box, no tunnel needed:

```bash
# Window 1: normal Claude (uses Anthropic API)
claude

# Window 2: local Qwen
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude --model openai/qwen-3.5-9b
```

You can run both side by side — use Claude for complex tasks, local Qwen for quick edits.

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

5. In Cursor Settings → Models:
   - Click **+ Add Custom Model** → `deepseek-r1-0528`
   - Override OpenAI Base URL → `https://<your-tunnel>.trycloudflare.com/v1`
   - API Key → any string (e.g. `sk-1234`)

6. Select `deepseek-r1-0528` in Chat mode (not Agent mode).

We use `deepseek-r1-0528` as the model name because Cursor's validation accepts it. LiteLLM maps it to your local Qwen server.

### Cline (VS Code extension — full agent mode)

Cline supports local models with no restrictions:

1. Install [Cline](https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev) in VS Code
2. Settings → API Provider → OpenAI Compatible
3. Base URL: `http://localhost:8080/v1`
4. Model: `qwen-3.5-9b`
5. API Key: any string

Works in agent mode — no tunnel, no name hacks.

### Continue (VS Code extension — autocomplete + chat)

Best for tab completion with local models:

1. Install [Continue](https://marketplace.visualstudio.com/items?itemName=Continue.continue) in VS Code
2. Configure `~/.continue/config.yaml` to point at `http://localhost:8080`

### Void (open source Cursor alternative)

[Void](https://voideditor.com/) is a VS Code fork with native local model support. No tunnels or workarounds needed.

## Model Selection Guide

| GPU VRAM | Recommended Model | Q4_K_M Size | Context |
|----------|-------------------|-------------|---------|
| 8GB | Qwen3.5-9B | 5.3GB | 16-32K |
| 12GB | Qwen3.5-9B | 5.3GB | 128K+ |
| 16GB | Qwen3.5-9B or Qwen3-14B | 5.3-8.4GB | 128K+ |
| 24GB | Qwen3.5-27B | ~16GB | 262K |

### Why Qwen3.5-9B?

Released March 2, 2026, part of the Qwen3.5 Small Series. Key advantages:
- Outperforms the previous-gen Qwen3-30B on most benchmarks despite being 9B
- 262K native context with Gated DeltaNet hybrid architecture
- Only 5.3GB in Q4_K_M — fits on any 8GB+ GPU with room for large context
- ~65 tok/s on RTX 4070 Ti, ~43 tok/s on RTX 3060

### Honest comparison to premium models

Qwen3.5-9B sits roughly in the **GPT-4o-mini / Claude Haiku tier**. It handles single-file tasks, completions, and simple bugs well. It struggles with complex multi-step reasoning and autonomous agentic workflows compared to Claude Sonnet/Opus or GPT-4o.

Best used for: fast completions, quick edits, boilerplate, explanations, unit tests.
Still need a premium model for: complex refactoring, multi-file changes, agentic "keep hammering" workflows.

## Troubleshooting

### OOM / CUDA out of memory
Reduce context: `-c 32768` or `-c 16384`. The KV cache scales with context size.

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
- [unsloth](https://huggingface.co/unsloth) for GGUF quantizations
- [LiteLLM](https://github.com/BerriAI/litellm) for the proxy workaround

## License

MIT
