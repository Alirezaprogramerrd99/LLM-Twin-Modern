# scripts/langchain_flow.ps1
$ErrorActionPreference = "Stop"

# --- Config ---
$ApiBase = "http://127.0.0.1:8000"
$Url     = "https://docs.langchain.com/oss/python/langchain/overview"

$SearchQ = "What is LangChain"
$AskQ    = "What is LangChain?"
$K       = 3

Write-Host "=== LangChain Web RAG Flow ==="
Write-Host "API: $ApiBase"
Write-Host "URL: $Url"
Write-Host "k  : $K"
Write-Host ""

# -------- 1) Ingest --------
Write-Host "=== 1) INGEST ==="
$bodyJson = (@{ url = $Url } | ConvertTo-Json -Depth 5)

try {
  $ingest = Invoke-RestMethod `
    -Uri "$ApiBase/ingest/url" `
    -Method Post `
    -Body $bodyJson `
    -ContentType "application/json"

  Write-Host "`n--- Ingest Response ---"
  $ingest | Format-List
} catch {
  Write-Host "`n[ERROR] Ingest failed:"
  throw
}

# -------- 2) Search --------
Write-Host "`n=== 2) SEARCH ==="
$encodedSearchQ = [System.Uri]::EscapeDataString($SearchQ)

try {
  $search = Invoke-RestMethod -Uri "$ApiBase/search?q=$encodedSearchQ&k=$K" -Method Get
  Write-Host "`n--- Search Results ---"
  $search | Format-Table -AutoSize
} catch {
  Write-Host "`n[ERROR] Search failed:"
  throw
}

# -------- 3) Ask --------
Write-Host "`n=== 3) ASK ==="
$encodedAskQ = [System.Uri]::EscapeDataString($AskQ)

try {
  $ask = Invoke-RestMethod -Uri "$ApiBase/ask?q=$encodedAskQ&k=$K" -Method Get
  Write-Host "`n--- Ask Response ---"
  $ask | Format-List
} catch {
  Write-Host "`n[ERROR] Ask failed:"
  throw
}

Write-Host "`n✅ Done."
