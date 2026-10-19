$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Virtual environment not found at $pythonPath"
}

Push-Location $projectRoot
try {
    & $pythonPath -m ruff check src tests
    & $pythonPath -m pytest
    & $pythonPath -m prism.ml.evaluate --config configs/training.json
    & $pythonPath -m prism.ml.inference --demo
}
finally {
    Pop-Location
}
