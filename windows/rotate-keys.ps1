<#
  Apply changes made to api-keys.txt.

  You only need this to REVOKE a key. Adding a device does not need it -- every
  key already listed in the file is valid, so hand a spare to the new device and
  it works immediately.

  llama-server reads the key file at startup only, so revoking means restarting.
  That costs ~40s of model reload and drops any in-flight session, which is why
  spares are pre-provisioned rather than minted on demand.
#>
param([switch]$Show)

$KeyFile = 'J:\llama\api-keys.txt'
$keys = Get-Content $KeyFile | Where-Object { $_.Trim() -and $_ -notmatch '^\s*#' }

if ($Show) {
  Write-Host "Valid keys ($($keys.Count)):" -ForegroundColor Cyan
  $keys | ForEach-Object { "  $_" }
  return
}

Write-Host "Restarting llama-server so $KeyFile takes effect..." -ForegroundColor Cyan
Write-Host "  $($keys.Count) key(s) will be valid." 
& 'J:\llama\serve-lan.ps1'
