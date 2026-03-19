$ErrorActionPreference = 'Stop'

$ProjectRoot = 'D:\0\PythonDemo\rebar_count'
$ApiDir = Join-Path $ProjectRoot 'rebar_mvp\counting-api'
$CondaExe = 'D:\miniconda\Scripts\conda.exe'
$CondaEnv = 'rebar_count'
$Port = 8000

if (-not (Test-Path $ApiDir)) {
  throw "API directory not found: $ApiDir"
}
if (-not (Test-Path $CondaExe)) {
  throw "conda.exe not found: $CondaExe"
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

$procIds = netstat -ano | Select-String ":$Port" | ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -Unique
foreach ($procId in $procIds) {
  try {
    Stop-Process -Id $procId -Force -ErrorAction Stop
    Write-Host "Stopped process on port ${Port}: $procId" -ForegroundColor Yellow
  } catch {
    Write-Host "Skip stopping process ${procId}: $($_.Exception.Message)" -ForegroundColor DarkYellow
  }
}

Write-Host "Starting H5/API on http://127.0.0.1:$Port ... (conda env: $CondaEnv)" -ForegroundColor Green
& $CondaExe run -n $CondaEnv python -m uvicorn app.main:app --host 127.0.0.1 --port $Port
