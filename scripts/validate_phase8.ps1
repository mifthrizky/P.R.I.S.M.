$ErrorActionPreference = "Stop"

Push-Location web
try {
    npm run lint
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    npm run build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}

docker compose config --quiet
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
