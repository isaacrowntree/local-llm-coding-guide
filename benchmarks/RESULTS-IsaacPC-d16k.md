## Local benchmark — NVIDIA GeForce RTX 4070 Ti (12282 MiB)

- **Date:** 2026-07-25
- **GPU / driver:** NVIDIA GeForce RTX 4070 Ti, driver 595.97, idle VRAM 536 MiB
- **llama.cpp:** rev `0893f50f2` via `llama-bench`
- **Flags:** `-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0`  •  prefill `-p 2048`, decode `-n 128` at depth `16384`, `-r 5`
- **Peak VRAM** = max total GPU memory in use during the run (includes ~536 MiB baseline).
- **Median, not mean.** This box shows high run-to-run GPU contention under WSL2 (the Windows host desktop intermittently grabs the GPU). Numbers are the **median** per-rep sample; the range column shows the spread. Decode is at depth `16384` — set `DEPTH=16384` for a realistic long-context number (much lower).

| Model | Disk | Prefill tok/s | Decode tok/s (median) | Decode range | Peak VRAM (MiB) | Notes |
|-------|------|--------------:|----------------------:|:------------:|----------------:|-------|
| Qwen3.6-35B-A3B-UD-IQ2_M.gguf | 11G | 3238 | 52 | 0-63 | 11930 |  |
| gemma-4-E4B-it-Q4_K_M.gguf | 4.7G | 6236 | 89 | 70-90 | 4298 |  |
| DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf | 4.7G | 2314 | 63 | 53-63 | 6226 |  |
| Qwen3.5-9B-Q4_K_M.gguf | 5.3G | 4029 | 70 | 69-70 | 6320 |  |
| Qwen3-14B-Q4_K_M.gguf | 8.4G | 1265 | 40 | 36-41 | 10055 |  |
| mistralai_Devstral-Small-2-24B-Instruct-2512-IQ2_M.gguf | 7.6G | 1051 | 43 | 38-43 | 9307 |  |

_Regenerate with `./benchmark.sh`. Numbers are for this exact GPU + llama.cpp build._
