$ErrorActionPreference = "Stop"

& .\.venv\Scripts\python.exe -m ruff check src tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& .\.venv\Scripts\python.exe -m pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& .\.venv\Scripts\python.exe -c "from prism.llm.client import MitigationGenerator; r=MitigationGenerator(timeout_seconds=0.1,max_attempts=1).generate('OSF',0.9,[{'document_id':'D','title':'T','page':1,'text':'x'}],True); assert r.status == 'MANUAL_REVIEW_REQUIRED'"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
