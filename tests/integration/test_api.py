from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from prism.api.app import create_app
from prism.api.runtime import PrismRuntime
from prism.llm.schemas import MitigationResponse

NORMAL_PAYLOAD = {
    "sensor_id": "CNC-Milling-01",
    "timestamp": "2026-08-31T08:00:00Z",
    "metrics": {
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551,
        "torque_nm": 42.8,
        "tool_wear_min": 21,
    },
}


class BinaryPipeline:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, features) -> np.ndarray:
        return np.asarray([[1 - self.probability, self.probability]])


class FailurePipeline:
    def predict_proba(self, features) -> np.ndarray:
        return np.asarray([[0.1, 0.1, 0.2, 0.9]])


class RecordingRetriever:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        self.calls += 1
        return [
            {
                "document_id": "OSHA-3120",
                "title": "Control of Hazardous Energy",
                "page": 7,
                "text": "Isolate energy and apply lockout devices.",
                "dense_score": 0.9,
                "bm25_score": 2.0,
            }
        ]


class RecordingGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, failure, confidence, evidence, evidence_is_sufficient):
        self.calls += 1
        return MitigationResponse.model_validate(
            {
                "status": "MITIGATION_READY",
                "predicted_failure": failure,
                "confidence_score": confidence,
                "mitigation_plan": {
                    "immediate_action": "Stop and isolate the machine.",
                    "required_ppe": ["Safety glasses"],
                    "safety_steps": ["Apply lockout devices."],
                    "citations": [
                        {
                            "document_id": "OSHA-3120",
                            "title": "Control of Hazardous Energy",
                            "page": 7,
                        }
                    ],
                },
                "reason": None,
            }
        )


def bundle(probability: float) -> dict:
    return {
        "pipeline": BinaryPipeline(probability),
        "failure_type_pipeline": FailurePipeline(),
        "metadata": {
            "decision_threshold": 0.5,
            "failure_type_columns": ["TWF", "HDF", "PWF", "OSF"],
            "failure_type_thresholds": {
                "TWF": 0.5,
                "HDF": 0.5,
                "PWF": 0.5,
                "OSF": 0.5,
            },
            "model_version": "test-model",
        },
    }


def test_health_and_readiness() -> None:
    runtime = PrismRuntime(bundle(0.1), RecordingRetriever(), RecordingGenerator())
    with TestClient(create_app(runtime)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready"}


def test_healthy_request_skips_retrieval_and_generation() -> None:
    retriever = RecordingRetriever()
    generator = RecordingGenerator()
    runtime = PrismRuntime(bundle(0.1), retriever, generator)
    with TestClient(create_app(runtime)) as client:
        response = client.post("/v1/predict", json=NORMAL_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "HEALTHY"
    assert body["mitigation"] is None
    assert body["machine_type_assumed"] is True
    assert body["timings"]["retrieval_ms"] is None
    assert retriever.calls == 0
    assert generator.calls == 0


def test_anomaly_runs_retrieval_and_generation() -> None:
    retriever = RecordingRetriever()
    generator = RecordingGenerator()
    runtime = PrismRuntime(bundle(0.9), retriever, generator)
    with TestClient(create_app(runtime)) as client:
        response = client.post("/v1/predict", json=NORMAL_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "CRITICAL_WARNING"
    assert body["primary_failure"] == "OSF"
    assert body["mitigation"]["status"] == "MITIGATION_READY"
    assert retriever.calls == 1
    assert generator.calls == 1


def test_invalid_payload_is_rejected() -> None:
    runtime = PrismRuntime(bundle(0.1), RecordingRetriever(), RecordingGenerator())
    invalid = {**NORMAL_PAYLOAD, "metrics": {**NORMAL_PAYLOAD["metrics"], "torque_nm": -1}}
    with TestClient(create_app(runtime)) as client:
        response = client.post("/v1/predict", json=invalid)

    assert response.status_code == 422
