<#
  Claude Code - normal (Anthropic cloud, your subscription).

  Explicitly clears the ANTHROPIC_* variables the local-LLM launcher sets, so a
  stale value in this shell can never silently redirect a cloud session at
  llama-server. First run may prompt for /login.
#>
param([Parameter(ValueFromRemainingArguments=$true)] $ClaudeArgs)

foreach ($v in 'ANTHROPIC_BASE_URL','ANTHROPIC_AUTH_TOKEN','ANTHROPIC_MODEL','ANTHROPIC_SMALL_FAST_MODEL','ANTHROPIC_API_KEY') {
  Remove-Item "Env:$v" -ErrorAction SilentlyContinue
}

Write-Host "Claude Code - Anthropic cloud" -ForegroundColor Cyan
& "$env:USERPROFILE\.local\bin\claude.exe" --dangerously-skip-permissions @ClaudeArgs
