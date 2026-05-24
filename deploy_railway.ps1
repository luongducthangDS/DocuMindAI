# DocuMind AI — Railway Deployment Script
# Usage: .\deploy_railway.ps1
# Prerequisites: railway login must be done first (run: railway login)

param(
    [string]$ProjectName = "documind-ai",
    [switch]$SkipVars   # skip setting env vars (useful for re-deploys)
)

Set-StrictMode -Off
$ErrorActionPreference = "Stop"

Write-Host "=== DocuMind AI — Railway Deploy ===" -ForegroundColor Cyan

# ── 1. Check .env ─────────────────────────────────────────────────────────────
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
if (-not $SkipVars) {
    Write-Host "`nSetting environment variables on Railway..." -ForegroundColor Yellow

    # Required: LLM API keys
    $requiredVars = @("GROQ_API_KEY", "GOOGLE_API_KEY")
    foreach ($key in $requiredVars) {
        if ($envVars.ContainsKey($key) -and $envVars[$key] -notmatch "^[gA]sk_xxx|^AIza") {
            $val = $envVars[$key]
            Write-Host "  Setting $key..." -NoNewline
            railway variables set "$key=$val" 2>&1 | Out-Null
            Write-Host " OK" -ForegroundColor Green
        } else {
            Write-Warning "  $key not found or is placeholder — skipping. Set it manually in Railway dashboard!"
        }
    }

    # Model & collection config — must match what was used to ingest data/chroma_db
    $modelVars = @{
        "EMBEDDING_MODEL"   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        "CHROMA_COLLECTION" = "documind_legal"
        "ENVIRONMENT"       = "production"
        "API_PORT"          = "8081"
        "API_BASE_URL"      = "http://127.0.0.1:8081"
        "INITIALIZE_RAG_ON_STARTUP" = "false"
        "ENABLE_RERANKER"   = "false"
        "PRIMARY_LLM"       = "groq/llama-3.3-70b-versatile"
        "FALLBACK_LLM"      = "gemini/gemini-1.5-flash"
    }
    foreach ($kv in $modelVars.GetEnumerator()) {
        Write-Host "  Setting $($kv.Key)..." -NoNewline
        railway variables set "$($kv.Key)=$($kv.Value)" 2>&1 | Out-Null
        Write-Host " OK" -ForegroundColor Green
    }

    # Optional: LangSmith tracing
    if ($envVars.ContainsKey("LANGCHAIN_API_KEY") -and $envVars["LANGCHAIN_API_KEY"] -notmatch "ls__xxx") {
        Write-Host "  Setting LangSmith vars..." -NoNewline
        railway variables set "LANGCHAIN_TRACING_V2=true" 2>&1 | Out-Null
        railway variables set "LANGCHAIN_API_KEY=$($envVars['LANGCHAIN_API_KEY'])" 2>&1 | Out-Null
        railway variables set "LANGCHAIN_PROJECT=documind-ai" 2>&1 | Out-Null
        Write-Host " OK" -ForegroundColor Green
    }

    # Optional: CORS origins (set your Railway domain here after first deploy)
    if ($envVars.ContainsKey("ALLOWED_ORIGINS") -and $envVars["ALLOWED_ORIGINS"] -ne "") {
        Write-Host "  Setting ALLOWED_ORIGINS..." -NoNewline
        railway variables set "ALLOWED_ORIGINS=$($envVars['ALLOWED_ORIGINS'])" 2>&1 | Out-Null
        Write-Host " OK" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "NOTE: Redis is optional. Without it, sessions use in-memory storage" -ForegroundColor DarkYellow
    Write-Host "      (sessions reset on container restart). To add Redis:" -ForegroundColor DarkYellow
    Write-Host "      1. Railway dashboard → Add service → Redis" -ForegroundColor DarkYellow
    Write-Host "      2. Copy the REDIS_URL and set it: railway variables set REDIS_URL=<url>" -ForegroundColor DarkYellow
}

# ── 4. Deploy ─────────────────────────────────────────────────────────────────
Write-Host "`nDeploying to Railway..." -ForegroundColor Yellow
Write-Host "Build will take 8-15 min (model pre-download in Docker image)" -ForegroundColor DarkYellow
railway up --detach

Write-Host ""
Write-Host "Deploy triggered!" -ForegroundColor Green
Write-Host "  Monitor build:  railway logs --tail" -ForegroundColor Cyan
Write-Host "  Dashboard:      https://railway.app/dashboard" -ForegroundColor Cyan
Write-Host ""
Write-Host "After deploy, test the API:" -ForegroundColor Yellow
Write-Host '  $url = (railway domain)' -ForegroundColor Gray
Write-Host '  Invoke-RestMethod "$url/api/v1/health"' -ForegroundColor Gray
Write-Host '  Invoke-RestMethod "$url/api/v1/query" -Method Post -Body ''{"query":"Luật doanh nghiệp 2020 quy định gì?","session_id":"test"}'' -ContentType "application/json"' -ForegroundColor Gray
