# DocuMind AI - open backend + frontend in separate windows
# Usage: .\start.ps1

$root = $PSScriptRoot

$backendCmd = @"
`$env:HF_HOME='$root\data\hf_cache'
`$env:HF_HUB_CACHE='$root\data\hf_cache'
`$env:SENTENCE_TRANSFORMERS_HOME='$root\data\hf_cache'
`$env:ANONYMIZED_TELEMETRY='false'
Set-Location '$root'
Write-Host 'Starting backend... (~25s to load models)' -ForegroundColor Cyan
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8081 --reload
"@

$frontendCmd = @"
Set-Location '$root\frontend'
Write-Host 'Starting frontend...' -ForegroundColor Cyan
npm run dev
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd

Write-Host ""
Write-Host "  Opened 2 terminal windows:" -ForegroundColor Green
Write-Host "  - Backend:  http://localhost:8081  (loading ~25s)" -ForegroundColor Cyan
Write-Host "  - Frontend: http://localhost:5174" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Wait for backend to print 'Application startup complete'" -ForegroundColor DarkGray
Write-Host "  then open http://localhost:5174" -ForegroundColor DarkGray
Write-Host ""
