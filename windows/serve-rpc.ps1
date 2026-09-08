<#
  Distributed inference: run a model too large for this GPU alone by borrowing
  memory from another machine (e.g. a Mac) over the network.

  This box is the COORDINATOR -- it holds the .gguf and drives generation. The
  remote machine runs rpc-server and contributes memory. Layers are split across
  both; every token crosses the network, so latency matters more than bandwidth.

  On the Mac first:
      cd ~/llama.cpp && git pull
      cmake -B build -DGGML_METAL=ON -DGGML_RPC=ON
      cmake --build build -j$(sysctl -n hw.ncpu)
      ./build/bin/rpc-server -H 0.0.0.0 -p 50052 -c
  ...and allow inbound 50052 on the Mac (System Settings > Network > Firewall).

  Then here:
      windows\serve-rpc.ps1 -Remote 192.168.5.42            # Q6_K, the practical max
      windows\serve-rpc.ps1 -Remote 192.168.5.42 -Model J:\models\...Q4_K_M.gguf

  CAVEATS, learned the hard way:
    * Both machines need COMPATIBLE llama.cpp builds. RPC has no version
      negotiation -- a mismatch fails in confusing ways. Rebuild both from the
      same commit if anything looks wrong.
    * The RPC backend is EXPERIMENTAL upstream.
    * Expect this to be SLOWER than either machine running a model that already
      fits. Distributed inference buys capacity, not speed. Benchmark it against
      claude-local.ps1 with a high -Ncmoe before believing it is worth the setup.
#>
param(
  [Parameter(Mandatory=$true)][string]$Remote,      # Mac IP
  [int]$RemotePort = 50052,
  [string]$Model = 'J:\models\Qwen3.6-35B-A3B-UD-Q6_K.gguf',
  [int]$Ctx = 65536,
  [int]$Port = 8080,
  [string]$KeyFile = 'J:\llama\api-keys.txt'
)

$ErrorActionPreference = 'Stop'
$exe = 'J:\llama\bin\llama-server.exe'
if (-not (Test-Path $Model)) { throw "Model not found: $Model" }

# Fail fast if the peer is not up -- otherwise llama-server dies mid-load with
# a much less obvious error.
$probe = Test-NetConnection -ComputerName $Remote -Port $RemotePort -WarningAction SilentlyContinue
if (-not $probe.TcpTestSucceeded) {
  throw "No rpc-server at ${Remote}:${RemotePort}. Start it on the Mac and allow the port."
}
Write-Host "RPC peer reachable at ${Remote}:${RemotePort}" -ForegroundColor Green

$key = (Get-Content $KeyFile | Where-Object { $_.Trim() -and $_ -notmatch '^\s*#' } | Select-Object -First 1).Trim()
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force

# No -ncmoe here: with RPC, layers that do not fit locally go to the PEER rather
# than to this machine's CPU. Mixing both split strategies makes results unreadable.
$srvArgs = @(
  '-m',$Model,'--host','0.0.0.0','--port',"$Port",
  '--rpc',"${Remote}:${RemotePort}",
  '-ngl','99','-c',"$Ctx",'-b','512','-ub','256','-fa','on','-np','1',
  '--cache-type-k','q4_0','--cache-type-v','q4_0',
  '--jinja','--reasoning-budget','0','-a','qwen3.6-local',
  '--api-key-file',$KeyFile
)
Write-Host "Starting coordinator: $(Split-Path $Model -Leaf) across local GPU + ${Remote}" -ForegroundColor Cyan
Start-Process -FilePath $exe -ArgumentList $srvArgs `
  -RedirectStandardOutput 'J:\llama\server.log' `
  -RedirectStandardError  'J:\llama\server.err' -WindowStyle Hidden
Write-Host "  watch: Get-Content J:\llama\server.err -Wait -Tail 20"
