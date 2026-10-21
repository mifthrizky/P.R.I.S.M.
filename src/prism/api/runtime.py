"""Load and orchestrate the local PRISM inference components."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from prism.api.schemas import PredictionRequest
from prism.llm.client import MitigationGenerator, retrieval_query
from prism.ml.artifacts import load_bundle
from prism.ml.inference import predict_payload
from prism.rag.index import DEFAULT_EMBEDDING_MODEL, E5Embedder
from prism.rag.retriever import HybridRetriever, has_sufficient_evidence


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]: ...


class Generator(Protocol):
    def generate(
        self,
        predicted_failure: str,
        confidence_score: float,
        evidence: list[dict[str, Any]],
        evidence_is_sufficient: bool,
    ) -> Any: ...


class PrismRuntime:
    def __init__(self, bundle: dict[str, Any], retriever: Retriever, generator: Generator) -> None:
        self.bundle = bundle
        self.retriever = retriever
        self.generator = generator

    def predict(self, request: PredictionRequest, request_id: str) -> dict[str, Any]:
        started = perf_counter()
        payload = request.model_dump(mode="json", exclude_none=True)

        ml_started = perf_counter()
        prediction = predict_payload(self.bundle, payload)
        ml_ms = _elapsed_ms(ml_started)
        retrieval_ms: float | None = None
        generation_ms: float | None = None
        mitigation = None

        if prediction["status"] == "CRITICAL_WARNING":
            failure_type = prediction["primary_failure"] or "UNKNOWN_FAILURE"
            retrieval_started = perf_counter()
            evidence = self.retriever.retrieve(retrieval_query(failure_type), top_k=5)
            sufficient = has_sufficient_evidence(evidence)
            retrieval_ms = _elapsed_ms(retrieval_started)

            generation_started = perf_counter()
            mitigation = self.generator.generate(
                failure_type,
                prediction["failure_probability"],
                evidence,
                evidence_is_sufficient=sufficient,
            )
            generation_ms = _elapsed_ms(generation_started)

        return {
            "request_id": request_id,
            **prediction,
            "mitigation": mitigation,
            "timings": {
                "ml_ms": ml_ms,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": _elapsed_ms(started),
            },
        }


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


def load_default_runtime(
    model_path: str | Path = "artifacts/model_bundle.joblib",
    knowledge_path: str | Path = "artifacts/knowledge",
) -> PrismRuntime:
    bundle = load_bundle(model_path)
    embedder = E5Embedder(DEFAULT_EMBEDDING_MODEL)
    retriever = HybridRetriever(knowledge_path, embedder)
    return PrismRuntime(bundle, retriever, MitigationGenerator())
