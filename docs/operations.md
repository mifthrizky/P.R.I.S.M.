# PRISM Operations Runbook

## Deployment profiles

PRISM can run directly in a Python virtual environment or as an API and Ollama stack
with Docker Compose. The first installation requires internet access for Python packages,
the E5 embedding model, container images, and Qwen weights. The cached runtime can then
operate without external inference services.

The container uses `requirements-runtime.txt`; training-only XGBoost and PDF ingestion
dependencies stay outside the production image. PyTorch is installed from its CPU wheel
index because GPU inference is isolated to Ollama.

## Local Python

```powershell
.\scripts\setup.ps1
$env:HF_HOME = "$PWD\.cache\huggingface"
.\.venv\Scripts\python.exe -m uvicorn prism.api.app:app --host 127.0.0.1 --port 8000
```

Start Ollama separately and ensure the model named by `PRISM_LLM_MODEL` is installed.
Use `/health` for liveness and `/ready` for model/index readiness.

## Docker Compose with NVIDIA GPU

Copy `.env.example` to `.env` only when values need overriding. Never commit `.env`.

```powershell
docker compose up --build -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/ready
```

The first start pulls Qwen into `ollama_models` and downloads the embedding model into
`huggingface_models`. Preserve these named volumes for offline restarts.

## Operational safety

- Treat `MANUAL_REVIEW_REQUIRED` as a hard stop for automated action.
- Do not execute generated text directly against machinery or PLCs.
- Only load the committed model artifact or another artifact from a trusted build.
- Review source-document revisions and rebuild the knowledge index after any update.
- Retain request IDs and stage timings in centralized logs; do not log secrets or raw
  proprietary sensor payloads without an approved retention policy.

## Verification and rollback

Run `scripts/validate_phase6.ps1` before a release and `scripts/validate_phase7.ps1`
after packaging changes. Releases are marked by Git tags. Roll back by deploying a prior
tag together with its matching model and knowledge artifacts; never mix artifact versions.
