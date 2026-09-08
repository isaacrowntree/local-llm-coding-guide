<#
  Distributed inference: run a model too large for this GPU alone by pooling
  memory with another machine (e.g. a Mac) over the network.

  This box is the COORDINATOR -- it holds the .gguf and drives generation. The
  remote machine runs rpc-server and contributes memory. Layers are split across
  both, so every token crosses the network: latency matters more than bandwidth.

  MAC SIDE: see MAC-RPC-SETUP.md in the repo. Short version:
      cd ~/llama.cpp && git pull
      cmake -B build -DGGML_METAL=ON -DGGML_RPC=ON
      cmake --build build -j$(sysctl -n hw.ncpu)
      codesign -s - -f build/bin/rpc-server        # else the firewall re-prompts
      ./build/bin/rpc-server -H 0.0.0.0 -p 50052 -c
      ipconfig getifaddr en0                       # <- the -Remote value

  USAGE
      windows\serve-rpc.ps1 -Remote 192.168.5.42                   # Q6_K (default)
      windows\serve-rpc.ps1 -Remote 192.168.5.42 -Quant Q5_K_M     # gentler fallback
      windows\serve-rpc.ps1 -Remote 192.168.5.42 -Ncmoe 4          # if the pool is still short

  WHY Q6 AND NOT Q5-SOLO-ON-MAC
      Measured 2026-09-08: Q5_K_M (26.5GB) on the 36GB Mac loaded only by evicting
      file cache and background apps. Splitting is gentler than that -- Q6 across
      both leaves the Mac ~16GB free because this box absorbs ~10GB of layers.
      If the Mac still evicts heavily during an RPC run, drop to -Quant Q5_K_M.

  CAVEATS
      * Peers need COMPATIBLE llama.cpp builds. RPC has no version negotiation;
        a mismatch fails confusingly. This box runs b10852.
      * The RPC backend is EXPERIMENTAL upstream.
      * Expect this to be SLOWER than a model that already fits one machine.
        Distributed inference buys capacity, not speed. Benchmark against
        claude-local.ps1 with a high -Ncmoe before concluding it is worth it.
#>
param(
  [Parameter(Mandatory=$true)][string]$Remote,
  [int]$RemotePort = 50052,
  [ValidateSet('Q6_K','Q5_K_M','Q4_K_M')][string]$Quant = 'Q6_K',
  [string]$Model,
  [int]$Ctx    = 65536,
  [int]$Ncmoe  = 0,          # 0 = none. With RPC, overflow goes to the PEER, not this CPU.
  [int]$Port   = 8080,
  [string]$KeyFile = 'J:\llama\api-keys.txt'
)

$ErrorActionPreference = 'Stop'
$exe = 'J:\llama\bin\llama-server.exe'

if (-not $Model) { $Model = "J:\models\Qwen3.6-35B-A3B-UD-$Quant.gguf" }

if (-not (Test-Path $Model)) {
  if (Test-Path "$Model.part") {
    $got = (Get-Item "$Model.part").Length / 1GB
    throw ("Model still downloading: {0:N1} GB so far at {1}.part" -f $got, $Model)
  }
  throw "Model not found: $Model"
}

# Fail fast on an absent peer. Without this llama-server dies mid-load with a
# much less obvious error.
Write-Host "Checking RPC peer ${Remote}:${RemotePort} ..." -NoNewline
$probe = Test-NetConnection -ComputerName $Remote -Port $RemotePort -WarningAction SilentlyContinue
if (-not $probe.TcpTestSucceeded) {
  Write-Host ""
  throw @"
No rpc-server at ${Remote}:${RemotePort}.
  * Is 'rpc-server -H 0.0.0.0 -p $RemotePort -c' running on the Mac?
  * Is it bound to 0.0.0.0 (not 127.0.0.1)?  lsof -nP -iTCP:$RemotePort -sTCP:LISTEN
  * macOS firewall authorised for the binary?  See MAC-RPC-SETUP.md
  * Same subnet? This box is 192.168.5.195/22.
"@
}
Write-Host " reachable." -ForegroundColor Green

$key = (Get-Content $KeyFile | Where-Object { $_.Trim() -and $_ -notmatch '^\s*#' } | Select-Object -First 1).Trim()
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force

$srvArgs = @(
  '-m',$Model,'--host','0.0.0.0','--port',"$Port",
  '--rpc',"${Remote}:${RemotePort}",
  '-ngl','99','-c',"$Ctx",'-b','512','-ub','256','-fa','on','-np','1',
  '--cache-type-k','q4_0','--cache-type-v','q4_0',
  '--jinja','--reasoning-budget','0','-a','qwen3.6-local',
  '--api-key-file',$KeyFile
)
if ($Ncmoe -gt 0) { $srvArgs += @('-ncmoe',"$Ncmoe") }

$sizeGB = (Get-Item $Model).Length / 1GB
Write-Host ("Loading {0} ({1:N1} GB) across local GPU + {2}" -f (Split-Path $Model -Leaf), $sizeGB, $Remote) -ForegroundColor Cyan
Write-Host "  Weights exceed local VRAM by design -- the remainder lives on the peer."
Write-Host "  Expect a slow first load: tensors ship over the network (rpc-server -c caches them)."

Start-Process -FilePath $exe -ArgumentList $srvArgs `
  -RedirectStandardOutput 'J:\llama\server.log' `
  -RedirectStandardError  'J:\llama\server.err' -WindowStyle Hidden

Write-Host ""
Write-Host "  progress : Get-Content J:\llama\server.err -Wait -Tail 20"
Write-Host "  local VRAM: nvidia-smi --query-gpu=memory.used --format=csv"
Write-Host "  ON THE MAC, watch memory pressure. If it evicts heavily, rerun with -Quant Q5_K_M."
Write-Host ""
Write-Host "  Benchmark once ready:"
Write-Host "    ./bench-claude-code.sh --base-url http://192.168.5.195:$Port --models qwen3.6-local"
