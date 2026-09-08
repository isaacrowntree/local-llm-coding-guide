<#
  Serve the local model to OTHER MACHINES ON YOUR LAN (e.g. a MacBook).

  Security posture:
    * --api-key-file : every request must present one of the keys listed in
                       api-keys.txt. Without it, anyone who can reach the port
                       can use your GPU and read your prompts. Multiple keys are
                       valid at once, so adding a device needs no restart --
                       the file is read at startup only.
    * --host 0.0.0.0 : required to accept non-loopback connections. On its own
                       this does NOT expose you to the internet -- your router
                       does not forward this port unless you tell it to. Do not
                       add a port-forward / DMZ entry for it.
    * The companion firewall rule is scoped to LocalSubnet, so only machines on
      your own network can connect. Run setup-lan-firewall.ps1 once (as admin).

  Traffic is plain HTTP: fine on a home LAN, not fine on a shared/public network.
  Use claude-local.ps1 instead when working on this machine -- it binds loopback
  only and needs no key.
#>
param(
  [int]$Ncmoe = 8,       # measured optimum on 12GB: 100 tok/s, 1.5GB free.
  [int]$Ctx   = 65536,   # 64k. Measured knee: 96k halves decode (48 tok/s); 128k
  [int]$Np    = 1,
  [int]$Reason = 0,
  [int]$Port  = 8080,
  [string]$KeyFile = 'J:\llama\api-keys.txt'
)

$ErrorActionPreference = 'Stop'
$exe   = 'J:\llama\bin\llama-server.exe'
$model = 'J:\models\Qwen3.6-35B-A3B-UD-IQ2_M.gguf'

if (-not (Test-Path $KeyFile)) { throw "API key file missing: $KeyFile" }
# Any key in the file is valid. Read the first non-comment line only to print
# a usable client command; the server itself accepts every listed key.
$key = (Get-Content $KeyFile | Where-Object { $_.Trim() -and $_ -notmatch '^\s*#' } | Select-Object -First 1).Trim()
if (-not $key) { throw "No keys found in: $KeyFile" }

Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force

$srvArgs = @(
  '-m',$model,'--host','0.0.0.0','--port',"$Port",
  '-ngl','99','-ncmoe',"$Ncmoe",'-c',"$Ctx",'-b','512','-ub','256',
  '-fa','on','-np',"$Np",'--cache-type-k','q4_0','--cache-type-v','q4_0',
  '--jinja','--reasoning-budget',"$Reason",'-a','qwen3.6-local',
  '--api-key-file',$KeyFile
)
Start-Process -FilePath $exe -ArgumentList $srvArgs `
  -RedirectStandardOutput 'J:\llama\server.log' `
  -RedirectStandardError  'J:\llama\server.err' -WindowStyle Hidden

# Report the LAN address clients should use (skip APIPA 169.254.* and virtual switches)
$ip = (Get-NetIPAddress -AddressFamily IPv4 |
       Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' -and $_.InterfaceAlias -notlike '*vEthernet*' } |
       Select-Object -First 1).IPAddress

Write-Host ""
Write-Host "llama-server listening on 0.0.0.0:$Port (LAN)" -ForegroundColor Cyan
Write-Host "  On your Mac:" -ForegroundColor Cyan
Write-Host ""
Write-Host "    export ANTHROPIC_BASE_URL=http://${ip}:$Port"
Write-Host "    export ANTHROPIC_AUTH_TOKEN=$key"
Write-Host "    export ANTHROPIC_MODEL=qwen3.6-local"
Write-Host "    claude --model qwen3.6-local --disallowedTools WebSearch WebFetch"
Write-Host ""
