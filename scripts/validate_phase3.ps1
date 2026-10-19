$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$env:HF_HOME = Join-Path $projectRoot ".cache\huggingface"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Virtual environment not found at $pythonPath"
}

Push-Location $projectRoot
try {
    & $pythonPath -m ruff check src tests
    if ($LASTEXITCODE -ne 0) { throw "Ruff validation failed" }
    & $pythonPath -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Pytest validation failed" }
    & $pythonPath -m prism.rag.evaluate --artifact-dir artifacts/knowledge `
        --evaluation documents/evaluation_queries.json --top-k 5
    if ($LASTEXITCODE -ne 0) { throw "Retrieval evaluation failed" }
}
finally {
    Pop-Location
}
