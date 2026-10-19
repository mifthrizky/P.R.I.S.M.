$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Virtual environment not found at $pythonPath"
}

Push-Location $projectRoot
try {
    & $pythonPath -m ruff check src tests
    if ($LASTEXITCODE -ne 0) { throw "Ruff validation failed" }
    & $pythonPath -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Pytest validation failed" }
    & $pythonPath -m prism.ml.evaluate --config configs/training.json
    if ($LASTEXITCODE -ne 0) { throw "Model evaluation failed" }
    & $pythonPath -m prism.ml.inference --demo
    if ($LASTEXITCODE -ne 0) { throw "Demo inference failed" }
}
finally {
    Pop-Location
}
