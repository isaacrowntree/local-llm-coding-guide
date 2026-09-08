<#
.SYNOPSIS
  Run Claude Code against the local Qwen3.6-35B-A3B model on this machine.

.DESCRIPTION
  Starts llama-server.exe (native Windows/CUDA) if it is not already running,
  then launches Claude Code pointed at it.

  llama.cpp serves the Anthropic Messages API directly at /v1/messages
  (PR #17570), so there is NO LiteLLM proxy in this path. Tool use needs --jinja.

  WebSearch/WebFetch are Anthropic *server-side* tools -- llama-server rejects
  them with a 400, and they cannot run against a local model. They are disabled.

.EXAMPLE
  .\claude-local.ps1
  .\claude-local.ps1 -- -p "fix the failing test"
#>
param(
  [int]$Ncmoe = 8,       # measured optimum on 12GB: 100 tok/s, 1.5GB free.
  [int]$Ctx   = 65536,   # 64k. Measured knee: 96k halves decode (48 tok/s); 128k
  [int]$Np     = 1,      # parallel slots. 1 is fine solo; forked subagents need one
                         # slot EACH or they queue and Claude Code reports
                         # "waiting for API response". 4 forks + main => -Np 5 -Ctx 204800.
                         # KV at q4_0 costs only ~287MB per 40960 tokens.
  [int]$Reason = 0,      # 0=off (best for agent loop), 1536~=medium, -1=max
  [Parameter(ValueFromRemainingArguments=$true)] $ClaudeArgs
)

$ErrorActionPreference = 'Stop'
$exe   = 'J:\llama\bin\llama-server.exe'
$model = 'J:\models\Qwen3.6-35B-A3B-UD-IQ2_M.gguf'
$port  = 8080

function Test-Server { try { $null = Invoke-RestMethod "http://127.0.0.1:$port/v1/models" -TimeoutSec 3; $true } catch { $false } }

if (Test-Server) {
  Write-Host "Reusing llama-server already on :$port" -ForegroundColor DarkGray
} else {
  $perSlot = [math]::Floor($Ctx / $Np)
  Write-Host "Starting llama-server (ncmoe=$Ncmoe ctx=$Ctx np=$Np => ${perSlot}/slot, reasoning=$Reason)..." -ForegroundColor Cyan
  if ($perSlot -lt 30000) { Write-Warning "Only $perSlot tokens per slot; Claude Code's prompt+tools is ~25K. Raise -Ctx." }
  $srvArgs = @(
    '-m',$model,'--host','127.0.0.1','--port',"$port",
    '-ngl','99','-ncmoe',"$Ncmoe",'-c',"$Ctx",'-b','512','-ub','256',
    '-fa','on','-np',"$Np",'--cache-type-k','q4_0','--cache-type-v','q4_0',
    '--jinja','--reasoning-budget',"$Reason",'-a','qwen3.6-local'
  )
  Start-Process -FilePath $exe -ArgumentList $srvArgs `
    -RedirectStandardOutput 'J:\llama\server.log' `
    -RedirectStandardError  'J:\llama\server.err' -WindowStyle Hidden

  $deadline = (Get-Date).AddSeconds(180)
  while (-not (Test-Server)) {
    if ((Get-Date) -gt $deadline) { throw "llama-server did not come up; see J:\llama\server.err" }
    Start-Sleep -Seconds 2
  }
  Write-Host "  ready." -ForegroundColor Green
}

$env:ANTHROPIC_BASE_URL          = "http://localhost:$port"
$env:ANTHROPIC_AUTH_TOKEN        = 'local'
$env:ANTHROPIC_MODEL             = 'qwen3.6-local'
$env:ANTHROPIC_SMALL_FAST_MODEL  = 'qwen3.6-local'
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue

Write-Host "Claude Code -> llama-server(:$port)  model qwen3.6-local" -ForegroundColor Cyan
& "$env:USERPROFILE\.local\bin\claude.exe" --dangerously-skip-permissions --disallowedTools WebSearch WebFetch @ClaudeArgs
