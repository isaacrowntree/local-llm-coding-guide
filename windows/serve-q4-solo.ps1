<#
  Config 1: Q4_K_M on THIS MACHINE ALONE, no RPC, no Mac.

  This is the control the RPC setup has to beat. Q4_K_M is 22.1GB against 12GB of
  VRAM, so most expert layers live on the CPU via -ncmoe. That sounds fatal but is
  not: Qwen3.6-35B-A3B activates only ~3B params per token, so CPU-side expert
  compute is modest, and transfers happen at PCIe speed rather than over Wi-Fi.

  Measured on IQ2_M for reference: -ncmoe 8 gave ~100 tok/s, -ncmoe 16 gave ~40.
  Q4 is roughly 2x the weights, so expect a much higher -ncmoe and lower tok/s.
  -Ncmoe 28 is a starting guess, not a measurement -- tune it.

  If this beats the RPC config, RPC is not worth the complexity. That is a
  legitimate and useful outcome.

  USAGE
      windows\serve-q4-solo.ps1                 # -Ncmoe 28
      windows\serve-q4-solo.ps1 -Ncmoe 24       # more on GPU: faster, more VRAM
      windows\serve-q4-solo.ps1 -Ncmoe 32       # more on CPU: slower, safer
#>
param(
  [int]$Ncmoe = 28,
  [int]$Ctx   = 65536,
  [int]$Port  = 8080,
  [string]$Model = 'J:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
  [string]$KeyFile = 'J:\llama\api-keys.txt'
)
$ErrorActionPreference = 'Stop'

if (-not (Test-Path $Model)) {
  if (Test-Path "$Model.part") {
    throw ("Still downloading: {0:N1} GB so far" -f ((Get-Item "$Model.part").Length/1GB))
  }
  throw "Model not found: $Model"
}

$key = (Get-Content $KeyFile | Where-Object { $_.Trim() -and $_ -notmatch '^\s*#' } | Select-Object -First 1).Trim()
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force

$srvArgs = @(
  '-m',$Model,'--host','0.0.0.0','--port',"$Port",
  '-ngl','99','-ncmoe',"$Ncmoe",'-c',"$Ctx",'-b','512','-ub','256',
  '-fa','on','-np','1','--cache-type-k','q4_0','--cache-type-v','q4_0',
  '--jinja','--reasoning-budget','0','-a','qwen3.6-local',
  '--api-key-file',$KeyFile
)
Write-Host ("Q4_K_M solo: -ncmoe {0}, ctx {1}" -f $Ncmoe, $Ctx) -ForegroundColor Cyan
Start-Process -FilePath 'J:\llama\bin\llama-server.exe' -ArgumentList $srvArgs `
  -RedirectStandardOutput 'J:\llama\server.log' `
  -RedirectStandardError  'J:\llama\server.err' -WindowStyle Hidden

Write-Host "  If it OOMs on load, raise -Ncmoe. If VRAM is left over, lower it."
Write-Host "  decode speed: Select-String 'eval time' J:\llama\server.err | Select-Object -Last 1"
