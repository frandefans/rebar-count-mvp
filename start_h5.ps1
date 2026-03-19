$ErrorActionPreference = 'Stop'

$ProjectRoot = 'D:\0\PythonDemo\rebar_count'
$ApiDir = Join-Path $ProjectRoot 'rebar_mvp\counting-api'
$CondaExe = 'D:\miniconda\Scripts\conda.exe'
$CondaEnv = 'rebar_count'
$PythonExe = 'D:\miniconda\envs\rebar_count\python.exe'
$Port = 8000
$HostIp = '0.0.0.0'

if (-not (Test-Path $ApiDir)) {
  throw "API directory not found: $ApiDir"
}
if (-not (Test-Path $CondaExe)) {
  throw "conda.exe not found: $CondaExe"
}
if (-not (Test-Path $PythonExe)) {
  throw "python.exe not found: $PythonExe"
}

Set-Location $ApiDir

$env:VISION_MODEL_PATH = 'D:\0\PythonDemo\rebar_count\model_recall_boost\model_final.pth'
$env:VISION_SCORE_THRESHOLD = '0.70'
$env:VISION_NMS_IOU = '0.20'
$env:VISION_ENABLE_TILE = 'true'
$env:VISION_TILE_SIZE = '1280'
$env:VISION_TILE_OVERLAP = '0.25'
$env:VISION_TILE_SCORE_DELTA = '0.00'
$env:VISION_CLUSTER_PAD = '0.10'
$env:VISION_PRE_NMS_TOPK = '3000'
$env:PYTHONNOUSERSITE = '1'

function Get-RealLanIps {
  $blocked = 'vpn|tun|tap|ppp|wireguard|wintun|tailscale|zerotier|vEthernet|hyper-v|virtualbox|vmware|loopback|wsl'
  try {
    $items = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop | ForEach-Object {
      $ip = $_.IPAddress
      $alias = $_.InterfaceAlias
      [PSCustomObject]@{ IPAddress = $ip; InterfaceAlias = $alias }
    }
    $items = $items | Where-Object {
      $_.IPAddress -notlike '127.*' -and
      $_.IPAddress -notlike '169.254.*' -and
      $_.InterfaceAlias -notmatch $blocked
    }
    return $items | Select-Object -ExpandProperty IPAddress -Unique
  } catch {
    $ips = ipconfig | Select-String 'IPv4|IPv4 地址|IPv4 Address' |
      ForEach-Object { (($_ -split ':')[-1]).Trim() } |
      Where-Object { $_ -and $_ -notlike '127.*' -and $_ -notlike '169.254.*' } |
      Select-Object -Unique
    return $ips
  }
}

$procIds = netstat -ano | Select-String ":$Port" | ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -Unique
foreach ($procId in $procIds) {
  try {
    Stop-Process -Id $procId -Force -ErrorAction Stop
    Write-Host "Stopped process on port ${Port}: $procId" -ForegroundColor Yellow
  } catch {
    Write-Host "Skip stopping process ${procId}: $($_.Exception.Message)" -ForegroundColor DarkYellow
  }
}

$lanIps = Get-RealLanIps

Write-Host "Starting H5/API on http://127.0.0.1:$Port ... (conda env: $CondaEnv)" -ForegroundColor Green
if ($lanIps -and $lanIps.Count -gt 0) {
  Write-Host "LAN URLs:" -ForegroundColor Cyan
  foreach ($ip in $lanIps) {
    Write-Host "  http://${ip}:$Port" -ForegroundColor Cyan
  }
} else {
  Write-Host "LAN URL not detected automatically. Please run: ipconfig" -ForegroundColor DarkYellow
}

& $PythonExe -m uvicorn app.main:app --host $HostIp --port $Port
