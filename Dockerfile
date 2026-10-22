FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface

WORKDIR /app

RUN groupadd --system prism && useradd --system --gid prism --create-home prism

COPY pyproject.toml requirements-runtime.txt README.md ./
RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements-runtime.txt

COPY src ./src
RUN python -m pip install --no-deps .

COPY artifacts ./artifacts
COPY documents ./documents
RUN mkdir -p /models/huggingface && chown -R prism:prism /app /models

USER prism
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "prism.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
