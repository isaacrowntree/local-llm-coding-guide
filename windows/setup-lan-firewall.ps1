<#
  One-time firewall rule allowing llama-server inbound from the LOCAL SUBNET ONLY.
  Must be run from an ELEVATED PowerShell (Run as administrator).

  -RemoteAddress LocalSubnet is the important part: machines outside your own
  subnet cannot reach the port even if something upstream tried to route to it.
  Combined with --api-key on the server, that is two independent controls.

  To undo:  Remove-NetFirewallRule -DisplayName 'llama-server (LAN only)'
#>
param([int]$Port = 8080)

$name = 'llama-server (LAN only)'

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
      ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Write-Error "Run this from an elevated PowerShell (Run as administrator)."
  exit 1
}

Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue | Remove-NetFirewallRule

New-NetFirewallRule -DisplayName $name `
  -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port `
  -RemoteAddress LocalSubnet `
  -Program 'J:\llama\bin\llama-server.exe' `
  -Profile Any `
  -Description 'Allow LAN clients (e.g. MacBook) to reach llama-server. Local subnet only.' | Out-Null

Write-Host "Created firewall rule '$name'" -ForegroundColor Green
Get-NetFirewallRule -DisplayName $name |
  Format-List DisplayName,Enabled,Direction,Action,Profile
Get-NetFirewallRule -DisplayName $name | Get-NetFirewallAddressFilter |
  Format-List RemoteAddress
