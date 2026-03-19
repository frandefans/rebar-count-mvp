$ErrorActionPreference = 'Continue'
$Port = 8000

function Get-RealLanIpInfo {
  $blocked = 'vpn|tun|tap|ppp|wireguard|wintun|tailscale|zerotier|vEthernet|hyper-v|virtualbox|vmware|loopback|wsl'
  try {
    return Get-NetIPAddress -AddressFamily IPv4 |
      Where-Object {
        $_.IPAddress -notlike '127.*' -and
        $_.IPAddress -notlike '169.254.*' -and
        $_.InterfaceAlias -notmatch $blocked
      } |
      Select-Object IPAddress, InterfaceAlias, PrefixLength
  } catch {
    return $null
  }
}

Write-Host '=== rebar_count LAN Check ===' -ForegroundColor Green
Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

Write-Host "`n[1] Local listening status (:${Port})" -ForegroundColor Yellow
netstat -ano | Select-String ":$Port"

Write-Host "`n[2] IPv4 addresses (filtered)" -ForegroundColor Yellow
try {
  $realLan = Get-RealLanIpInfo
  if ($realLan) {
    $realLan
  } else {
    ipconfig | Select-String 'IPv4|IPv4 地址|IPv4 Address'
  }
} catch {
  ipconfig | Select-String 'IPv4|IPv4 地址|IPv4 Address'
}

Write-Host "`n[3] Network profile" -ForegroundColor Yellow
try {
  Get-NetConnectionProfile | Select-Object Name, NetworkCategory, IPv4Connectivity
} catch {
  Write-Host 'Cannot query network profile (may need admin).' -ForegroundColor DarkYellow
}

Write-Host "`n[4] Firewall rules (8000)" -ForegroundColor Yellow
netsh advfirewall firewall show rule name="rebar_count_h5_8000"
netsh advfirewall firewall show rule name="rebar_count_h5_8000_any"

Write-Host "`n[5] Local HTTP test" -ForegroundColor Yellow
try {
  $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -UseBasicParsing -TimeoutSec 5
  Write-Host "127.0.0.1:$Port => $($r.StatusCode)" -ForegroundColor Green
} catch {
  Write-Host "127.0.0.1:$Port => FAIL ($($_.Exception.Message))" -ForegroundColor Red
}

Write-Host "`n[6] Suggestions" -ForegroundColor Yellow
Write-Host '1) Ensure start_h5.bat window remains open.'
Write-Host '2) Access using one of printed LAN URLs from start_h5.ps1.'
Write-Host '3) If NetworkCategory is Public, switch to Private in admin PowerShell.'
Write-Host '4) If VPN is enabled, use the filtered LAN IP and enable Allow LAN access / Split tunneling.'
Write-Host '5) On router/AP, disable AP Isolation / Client Isolation.'
