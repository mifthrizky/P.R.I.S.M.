"""FastAPI application for predictive maintenance orchestration."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from prism.api.runtime import PrismRuntime, load_default_runtime
from prism.api.schemas import PredictionRequest, PredictionResponse

LOGGER = logging.getLogger("prism.api")


def create_app(runtime: PrismRuntime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if runtime is not None:
            app.state.runtime = runtime
            app.state.startup_error = None
        else:
            try:
                app.state.runtime = load_default_runtime()
                app.state.startup_error = None
            except Exception as exc:  # readiness reports details without killing liveness
                LOGGER.exception("PRISM runtime failed to initialize")
                app.state.runtime = None
                app.state.startup_error = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(
        title="PRISM Predictive Maintenance API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request) -> dict[str, str]:
        if request.app.state.runtime is None:
            raise HTTPException(
                status_code=503,
                detail={"status": "not_ready", "reason": request.app.state.startup_error},
            )
        return {"status": "ready"}

    @app.post("/v1/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> dict[str, Any]:
        active_runtime = request.app.state.runtime
        if active_runtime is None:
            raise HTTPException(status_code=503, detail="PRISM runtime is not ready")
        request_id = str(uuid4())
        result = active_runtime.predict(payload, request_id)
        LOGGER.info(
            "prediction_complete request_id=%s sensor_id=%s status=%s total_ms=%s",
            request_id,
            payload.sensor_id,
            result["status"],
            result["timings"]["total_ms"],
        )
        return result

    return app


app = create_app()


def run() -> None:
    uvicorn.run("prism.api.app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    run()
