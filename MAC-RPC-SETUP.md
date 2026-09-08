# Mac side: distributed inference peer (RPC) + local server

**Audience:** a Claude Code session running on the Mac (M3 Pro, 36GB).
Everything below is meant to be executed on the **Mac**. The Windows box is
already configured — nothing on it needs changing.

## What we are building and why

The Windows box has an RTX 4070 Ti with **12GB VRAM**. That fits
Qwen3.6-35B-A3B only at **IQ2_M (2.7 bits/weight)** — the quantisation level where
quality damage starts to bite, especially for tool-call formatting in agentic
coding.

The Mac has **36GB unified memory**. Combined (~38GB usable) the two machines can
hold a far better quant. llama.cpp's RPC backend splits one model's layers across
machines over the network to make that possible.

**Three configurations to compare.** Do not assume RPC wins — the point is to
measure:

| # | Config | Size | Where | Prediction |
|---|--------|-----:|-------|-----------|
| 1 | Q4_K_M, PC alone, high `-ncmoe` (experts on CPU) | 22.1 GB | Windows | Likely fastest |
| 2 | Q4_K_M, Mac alone, Metal | 22.1 GB | **Mac** | Mac baseline |
| 3 | ~~Q5_K_M, Mac alone~~ | 26.5 GB | Mac | **TESTED — rejected, see below** |
| 4 | Q6_K, PC coordinator + Mac RPC peer | 29.3 GB | both | Highest quality; now the main event |

### Measured 2026-09-08: Q5_K_M does not fit the Mac comfortably

Q5_K_M **loaded, but only by evicting everything else** — file cache and
background apps pushed out. It technically runs; the machine does not feel usable
and performance is one page-fault storm away from collapsing. Compare the earlier
finding on this same Mac: the dense 31B swap-thrashed to **~6 tok/s**, 5x slower
than a model that fit. Swap, not OOM, is the failure mode here.

**So the Mac's solo ceiling is Q4_K_M (22.1GB), not Q5.**

**This makes RPC *more* attractive, not less** — which is counterintuitive enough
to state plainly:

| Config | Mac holds | Mac free for macOS |
|---|---:|---:|
| Q5_K_M solo on Mac | 26.5 GB | ~8 GB — evicts everything |
| Q6_K split PC + Mac | ~20 GB | **~16 GB** |

Splitting keeps *both* machines under their comfort ceiling, because the PC
absorbs ~10GB of layers. A **larger** model over RPC can be gentler on the Mac
than a smaller one running solo. That is the case RPC now has to prove.

Watch memory pressure during the RPC run too (`memory_pressure`, or Activity
Monitor's swap figure). If the peer's share still evicts heavily, cap it — see
`-d`/device selection on `rpc-server` — or drop to Q5_K_M split rather than Q6.

RPC pays **per-token network latency** — every token crosses the wire. Config 1
pays only PCIe-speed CPU transfers. A 3B-active MoE tolerates CPU-offloaded
experts very well (measured: `-ncmoe 8` gives ~100 tok/s on the PC). So config 1
beating config 3 is the expected outcome, and that is a legitimate result.

## Facts you need

| | |
|---|---|
| Windows box LAN IP | `192.168.5.195` |
| Windows llama-server | port `8080`, Anthropic API at `/v1/messages` |
| RPC port (Mac listens, PC dials in) | `50052` |
| API key | in `~/.llama-lan.conf` as `LLAMA_KEY`; ask the user if absent |
| Model repo | `unsloth/Qwen3.6-35B-A3B-GGUF` |

llama-server serves the **Anthropic Messages API natively** since llama.cpp
PR #17570. **There is no LiteLLM proxy in any of this.** Tool use requires
`--jinja`.

## Step 1 — Rebuild llama.cpp with RPC and Metal

The Mac's build may predate both the Anthropic endpoint (Jan 2026) and RPC.
Check first:

```bash
~/llama.cpp/build/bin/llama-server --version   # want a build number >= ~8700
```

Rebuild either way — `start-server-mac.sh` in this repo now passes `-DGGML_RPC=ON`,
but only when the binary is **missing**, not when it is stale:

```bash
cd ~/llama.cpp && git pull
cmake -B build -DGGML_METAL=ON -DGGML_RPC=ON
cmake --build build -j$(sysctl -n hw.ncpu)
ls build/bin/rpc-server        # must exist before continuing
```

> **Version matching matters.** RPC has no version negotiation between peers;
> a mismatch fails in confusing ways. The Windows box runs **b10852**. If RPC
> misbehaves, check `--version` on both and rebuild from the same commit.

## Step 2 — Config 2 first (Mac alone, Q4). Do this before any RPC work.

It is the useful baseline and needs no networking.

```bash
huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --local-dir ~/models      # 22.1GB
huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q5_K_M.gguf --local-dir ~/models      # 26.5GB, for config 3

REASONING=0 CTX=65536 ./start-server-mac.sh qwen3.6-35b-a3b
```

Q5_K_M solo has already been tested and rejected (see above) — do not repeat it.
Q4_K_M is the Mac's solo configuration.

Then, from a second terminal in this repo:

```bash
./bench-claude-code.sh --base-url http://localhost:8080 --models qwen3.6-local
```

Record: pass rate, wall time per task, `eff_tok_per_s`. Results land in
`bench/results/<timestamp>/`.

**Settings that matter, and why** (measured on the Windows box, 2026-09-08):

- `--reasoning-budget 0` — thinking tokens re-prefill every turn. Leaving this on
  is a large invisible tax on an agent loop. `start-server-mac.sh` defaults to
  `-1` (unlimited), so **you must override it**.
- `-c 65536` — 64K is a sharp knee. On the PC, 96K *halved* decode (100 → 48 tok/s)
  for context you rarely use. Re-measure on Metal; do not assume it transfers.
- `--jinja` — without it the Anthropic endpoint cannot emit tool calls and every
  agentic task fails. `start-server-mac.sh` now passes it.
- `-a qwen3.6-local` — without an alias `claude --model` cannot resolve.
- Do **not** pass `--bench` here. That flag pins small batches (`-b 512 -ub 256`)
  that exist to fit 12GB of VRAM; on 36GB of unified memory they only cost
  prefill throughput. Use `--bench` *only* when producing numbers meant to be
  compared directly against the NVIDIA runs.

## Step 3 — Config 3 (RPC peer)

Start the peer. `-c` enables the local tensor cache, which avoids re-shipping
weights across the network on every start:

```bash
~/llama.cpp/build/bin/rpc-server -H 0.0.0.0 -p 50052 -c
```

### Firewall on macOS

macOS's Application Firewall (ALF) is **per-application, not per-port** — there is
no "open port 50052" rule to add. You authorise the `rpc-server` binary itself.

**First check whether it is even on.** On many Macs it is off, in which case there
is nothing to do:

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
```

- `disabled` → **nothing to configure.** Skip to verification.
- `enabled` → authorise the binary:

```bash
FW=/usr/libexec/ApplicationFirewall/socketfilterfw
BIN=$HOME/llama.cpp/build/bin/rpc-server

sudo "$FW" --add "$BIN"
sudo "$FW" --unblockapp "$BIN"
sudo "$FW" --listapps | grep -A2 rpc-server     # confirm it says "Allow"
```

> **The dialog trap — this matters if you are an agent.** With the firewall on,
> macOS shows a GUI prompt: *"Do you want the application to accept incoming
> network connections?"* Nothing arrives until a human clicks Allow, and a
> headless session cannot click it. Running `--add`/`--unblockapp` **before**
> first launch pre-authorises the binary and suppresses the dialog. Do it in that
> order.
>
> **Unsigned binaries re-prompt on every rebuild.** A locally compiled
> `rpc-server` is unsigned, so its identity changes each time you rebuild and the
> authorisation does not stick. Ad-hoc sign it once after building:
>
> ```bash
> codesign -s - -f "$BIN"
> ```
>
> Re-run `--add`/`--unblockapp` after any rebuild that changes the binary.

**Stealth mode** (`--getstealthmode`) drops ICMP, so `ping` from the PC fails even
when the port is genuinely open. Do not use ping to test — use the port check
below. Leave stealth mode as you found it.

You do **not** need to touch `pf`. ALF is sufficient, and editing `pf.conf` on a
laptop that moves between networks causes more problems than it solves.

### Verify before involving the PC

```bash
ipconfig getifaddr en0                       # Wi-Fi IP; en1 if wired. Give this to the user.
lsof -nP -iTCP:50052 -sTCP:LISTEN            # must show rpc-server LISTEN on *:50052
nc -vz 127.0.0.1 50052                       # loopback: proves the process is up
```

If `lsof` shows `127.0.0.1:50052` rather than `*:50052`, you forgot `-H 0.0.0.0`
and no remote machine can reach it.

**Confirm you are on the same subnet as the PC.** The Windows box is
`192.168.5.195/22`, so the Mac must be within `192.168.4.0`–`192.168.7.255`:

```bash
ipconfig getifaddr en0
netstat -rn | grep default
```

A Mac on a *guest* SSID, a different band with client isolation, or a VPN will
look connected but be unable to reach the PC. This is the most common cause of a
silent failure here.

**Ask the user to confirm from the Windows side** before going further:

```powershell
Test-NetConnection -ComputerName <mac-ip> -Port 50052
```

`TcpTestSucceeded : True` means the path is good. `serve-rpc.ps1` runs this same
check and refuses to start without it, so a clean failure there points at the
firewall or the subnet, not at llama.cpp.

**Then tell the user the Mac's IP.** They run this on Windows:

```powershell
windows\serve-rpc.ps1 -Remote <mac-ip>
```

The PC dials the Mac, loads Q6_K (29.3GB) split across both, and serves on
`192.168.5.195:8080` as usual. Benchmark it the same way:

```bash
./bench-claude-code.sh --base-url http://192.168.5.195:8080 --models qwen3.6-local
```

(The harness passes the key from the environment; export `LLAMA_KEY` or use
`./connect-lan-mac.sh` for interactive work.)

## Gotchas already hit on the Windows side

- **A stray `llama-server` will hijack the port.** Recent builds default to a
  *router* mode, so an argument-less `llama-server` binds :8080 and answers every
  request with `400: model '<name>' not found`. If you see that, look for a second
  process before debugging anything else.
- **`--api-key-file` is read at startup only.** Verified: appending a key returns
  401 until restart. Multiple keys are valid at once, which is why spares exist.
- **`WebSearch`/`WebFetch` must be disabled.** They are Anthropic *server-side*
  tools; llama-server rejects them with a 400 and they cannot run locally anyway.
  All the launcher scripts pass `--disallowedTools WebSearch WebFetch`.
- **Cold start is ~45s.** Claude Code sends a ~21–25K token system prompt that
  must prefill before the first token. That is expected, not a hang.

## Reverse direction: if the Mac ever serves the PC

Everything above has the **PC dialling the Mac**. If you instead want the Mac to
serve models to other machines (the Mac running `llama-server`, the PC as client),
the firewall work is the same shape but on port `8080`, and you must also:

- bind with `--host 0.0.0.0` (the default `127.0.0.1` accepts nothing remote), and
- set `--api-key-file` — an unauthenticated model server on a LAN lets anyone who
  can reach the port use your machine and read your prompts.

`start-server-mac.sh` binds `0.0.0.0` already but sets **no API key**, so do not
expose it beyond loopback without adding one.

## Quick reference

```bash
# build
cd ~/llama.cpp && git pull
cmake -B build -DGGML_METAL=ON -DGGML_RPC=ON && cmake --build build -j$(sysctl -n hw.ncpu)
codesign -s - -f build/bin/rpc-server

# firewall (only if --getglobalstate says enabled)
FW=/usr/libexec/ApplicationFirewall/socketfilterfw
sudo $FW --add ~/llama.cpp/build/bin/rpc-server
sudo $FW --unblockapp ~/llama.cpp/build/bin/rpc-server

# run peer
~/llama.cpp/build/bin/rpc-server -H 0.0.0.0 -p 50052 -c

# verify + hand the IP to the user
ipconfig getifaddr en0
lsof -nP -iTCP:50052 -sTCP:LISTEN
```

## What to report back

For each config: pass rate (n/4), per-task wall time, `eff_tok_per_s`, peak memory,
and for RPC whether it was stable. Commit results to `bench/results/` and note the
llama.cpp build number on each machine — numbers without build numbers are not
reproducible.

If RPC turns out slower than config 1, **say so plainly**. That is the finding, and
this repo exists to measure rather than assume.
