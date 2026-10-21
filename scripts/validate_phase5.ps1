$ErrorActionPreference = "Stop"

& .\.venv\Scripts\python.exe -m ruff check src tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& .\.venv\Scripts\python.exe -m pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& .\.venv\Scripts\python.exe -c "from prism.api.app import app; assert app.title == 'PRISM Predictive Maintenance API'"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
