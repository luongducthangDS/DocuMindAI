# DocuMind AI — Railway Deployment Script
# Usage: .\deploy_railway.ps1
# Prerequisites: railway login must be done first

param(
    [string]$ProjectName = "documind-ai"
)

Set-StrictMode -Off
$ErrorActionPreference = "Stop"

Write-Host "=== DocuMind AI — Railway Deploy ===" -ForegroundColor Cyan

# ── 1. Read .env ──────────────────────────────────────────────────────────────
if (-not (Test-Path ".env")) {
    Write-Error ".env file not found. Cannot deploy without API keys."
    exit 1
}

$envVars = @{}
Get-Content ".env" | ForEach-Object {
    if ($_ -match "^\s*([^#=]+)=(.*)$") {
        $envVars[$Matches[1].Trim()] = $Matches[2].Trim()
    }
}

# ── 2. Check Railway login ────────────────────────────────────────────────────
Write-Host "`nChecking Railway auth..." -ForegroundColor Yellow
$whoami = railway whoami 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Not logged in. Run: railway login" -ForegroundColor Red
    exit 1
}
Write-Host "Logged in as: $whoami" -ForegroundColor Green

# ── 3. Set environment variables ──────────────────────────────────────────────
Write-Host "`nSetting environment variables on Railway..." -ForegroundColor Yellow

$varsToSet = @(
    "GROQ_API_KEY",
    "GOOGLE_API_KEY",
    "EMBEDDING_MODEL",
    "CHROMA_COLLECTION"
)

foreach ($key in $varsToSet) {
    if ($envVars.ContainsKey($key)) {
        $val = $envVars[$key]
        Write-Host "  Setting $key..." -NoNewline
        railway variables set "$key=$val" 2>&1 | Out-Null
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Warning "  $key not found in .env — skipping"
    }
}

# Always set production environment
railway variables set "ENVIRONMENT=production" 2>&1 | Out-Null
Write-Host "  Setting ENVIRONMENT=production... OK" -ForegroundColor Green

# ── 4. Deploy ─────────────────────────────────────────────────────────────────
Write-Host "`nDeploying to Railway (this may take 5-10 min for image build)..." -ForegroundColor Yellow
railway up --detach

Write-Host "`nDeploy triggered. Monitor at: https://railway.app/dashboard" -ForegroundColor Cyan
Write-Host "Check logs: railway logs --tail" -ForegroundColor Cyan
