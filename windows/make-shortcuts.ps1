$desktop = [Environment]::GetFolderPath('Desktop')
$ps      = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$icon    = "$env:USERPROFILE\.local\bin\claude.exe,0"
$w       = New-Object -ComObject WScript.Shell

function New-Lnk($name, $script, $desc, $workdir) {
  $lnk = $w.CreateShortcut((Join-Path $desktop "$name.lnk"))
  $lnk.TargetPath       = $ps
  $lnk.Arguments        = "-NoExit -NoProfile -ExecutionPolicy Bypass -File `"$script`""
  $lnk.WorkingDirectory = $workdir
  $lnk.IconLocation     = $icon
  $lnk.Description      = $desc
  $lnk.Save()
  Write-Output "created: $name.lnk"
}

New-Lnk 'Claude Code' 'J:\llama\claude-cloud.ps1' `
  'Claude Code against the Anthropic cloud (your subscription)' $env:USERPROFILE

New-Lnk 'Claude Code (Local LLM)' 'J:\llama\claude-local.ps1' `
  'Claude Code against local Qwen3.6-35B-A3B via llama-server on :8080' $env:USERPROFILE
