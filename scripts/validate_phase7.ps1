$ErrorActionPreference = "Stop"

& .\.venv\Scripts\python.exe -m ruff check src tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& .\.venv\Scripts\python.exe -m pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker compose config --quiet
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& .\.venv\Scripts\python.exe -c "from prism.api.app import app; assert app.version == '0.1.0'"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
