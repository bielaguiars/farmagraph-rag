# Sobe Neo4j (Docker), Ollama e a interface Streamlit do FarmaGraph RAG.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "1/3 Neo4j..." -ForegroundColor Cyan
docker compose up -d

Write-Host "2/3 Ollama..." -ForegroundColor Cyan
try { Invoke-RestMethod http://localhost:11434/api/version -TimeoutSec 3 | Out-Null }
catch { Start-Process -WindowStyle Hidden ollama -ArgumentList "serve"; Start-Sleep -Seconds 3 }

Write-Host "Aguardando o Neo4j aceitar conexões..." -ForegroundColor Cyan
for ($i = 0; $i -lt 30; $i++) {
  try { Invoke-WebRequest http://localhost:7474 -UseBasicParsing -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep -Seconds 2 }
}

Write-Host "3/3 Interface em http://localhost:8501" -ForegroundColor Green
& .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
