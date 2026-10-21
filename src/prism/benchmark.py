"""Deterministic acceptance and latency benchmark for PRISM orchestration."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from prism.api.runtime import PrismRuntime
from prism.api.schemas import PredictionRequest, PredictionResponse
from prism.llm.schemas import MitigationResponse
from prism.ml.artifacts import load_bundle
from prism.ml.inference import DEMO_PAYLOADS


class BenchmarkRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return [
            {
                "document_id": "OSHA-3120",
                "title": "Control of Hazardous Energy (Lockout/Tagout)",
                "page": 7,
                "text": "Isolate energy and apply lockout devices before servicing.",
                "dense_score": 0.9,
                "bm25_score": 2.0,
            }
        ]


class BenchmarkGenerator:
    """Schema-valid deterministic stand-in; it does not estimate live Ollama latency."""

    def generate(self, failure, confidence, evidence, evidence_is_sufficient):
        return MitigationResponse.model_validate(
            {
                "status": "MITIGATION_READY",
                "predicted_failure": failure,
                "confidence_score": confidence,
                "mitigation_plan": {
                    "immediate_action": "Stop and isolate the machine before inspection.",
                    "required_ppe": ["Safety glasses"],
                    "safety_steps": ["Apply approved lockout devices."],
                    "citations": [
                        {
                            "document_id": evidence[0]["document_id"],
                            "title": evidence[0]["title"],
                            "page": evidence[0]["page"],
                        }
                    ],
                },
                "reason": None,
            }
        )


def percentiles(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(float(np.percentile(values, 50)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "p99_ms": round(float(np.percentile(values, 99)), 3),
    }


def run_benchmark(runs: int) -> dict[str, Any]:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    bundle = load_bundle("artifacts/model_bundle.joblib")
    runtime = PrismRuntime(bundle, BenchmarkRetriever(), BenchmarkGenerator())
    normal = PredictionRequest.model_validate(DEMO_PAYLOADS[0])
    anomaly = PredictionRequest.model_validate(DEMO_PAYLOADS[1])
    normal_latencies: list[float] = []
    anomaly_latencies: list[float] = []
    schema_successes = 0

    for index in range(runs):
        started = perf_counter()
        normal_result = runtime.predict(normal, f"normal-{index}")
        normal_latencies.append((perf_counter() - started) * 1000)
        PredictionResponse.model_validate(normal_result)

        started = perf_counter()
        anomaly_result = runtime.predict(anomaly, f"anomaly-{index}")
        anomaly_latencies.append((perf_counter() - started) * 1000)
        parsed = PredictionResponse.model_validate(anomaly_result)
        if parsed.mitigation is not None:
            schema_successes += 1

    metadata = bundle["metadata"]
    rag_report = json.loads(
        Path("artifacts/knowledge/evaluation_report.json").read_text(encoding="utf-8")
    )
    roc_auc = float(metadata["metrics"]["holdout"]["roc_auc"])
    recall_at_5 = float(rag_report["recall_at_k"])
    schema_rate = schema_successes / runs
    anomaly_stats = percentiles(anomaly_latencies)
    gates = {
        "roc_auc_at_least_0_88": roc_auc >= 0.88,
        "rag_recall_at_5_is_1": recall_at_5 == 1.0,
        "structured_output_rate_is_1": schema_rate == 1.0,
        "deterministic_orchestration_p95_under_3s": anomaly_stats["p95_ms"] < 3000,
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "deterministic_orchestration",
        "limitations": (
            "Generation uses a schema-valid deterministic stand-in. This report does not "
            "claim live Ollama/Qwen latency; run a live benchmark after Ollama is available."
        ),
        "runs_per_branch": runs,
        "ml_test_roc_auc": roc_auc,
        "rag_recall_at_5": recall_at_5,
        "structured_output_success_rate": schema_rate,
        "normal_latency": percentiles(normal_latencies),
        "anomaly_latency": anomaly_stats,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("artifacts/acceptance_report.json"))
    args = parser.parse_args()
    report = run_benchmark(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
