$ErrorActionPreference = "Stop"

& .\.venv\Scripts\python.exe -m ruff check src tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& .\.venv\Scripts\python.exe -m pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& .\.venv\Scripts\python.exe -m prism.benchmark --runs 100 --output artifacts/acceptance_report.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
