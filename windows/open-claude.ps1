# Opens a new PowerShell window running Claude Code against the local model.
# Reuses claude-local.ps1, which starts llama-server if it is not already up.
Start-Process -FilePath 'powershell.exe' `
  -ArgumentList '-NoExit','-NoProfile','-ExecutionPolicy','Bypass','-File','J:\llama\claude-local.ps1' `
  -WorkingDirectory 'J:\llama\demo' `
  -WindowStyle Normal
Write-Output 'Claude Code window launched'
