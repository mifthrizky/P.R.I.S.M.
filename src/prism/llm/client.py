"""Ollama client that only returns validated and grounded mitigation JSON."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import ValidationError

from prism.llm.schemas import MitigationResponse

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

FAILURE_QUERIES = {
    "TWF": "tool wear maintenance machine shutdown lockout tagout PPE",
    "HDF": "machine overheating heat hazard shutdown lockout tagout PPE",
    "PWF": "power failure electrical hazardous energy isolation lockout tagout PPE",
    "OSF": "machine overstrain overload rotating parts shutdown lockout tagout PPE",
}


def retrieval_query(failure_type: str) -> str:
    return FAILURE_QUERIES.get(
        failure_type,
        "unexpected machine failure safe shutdown hazardous energy lockout tagout PPE",
    )


def manual_review(
    predicted_failure: str, confidence_score: float, reason: str
) -> MitigationResponse:
    return MitigationResponse(
        status="MANUAL_REVIEW_REQUIRED",
        predicted_failure=predicted_failure,
        confidence_score=confidence_score,
        reason=reason,
    )


def _evidence_payload(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "document_id": item["document_id"],
            "title": item["title"],
            "page": item["page"],
            "text": item["text"],
        }
        for item in evidence
    ]


def _messages(
    predicted_failure: str,
    confidence_score: float,
    evidence: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system = (
        "You are an industrial safety decision-support component. Use only the supplied "
        "evidence. Never invent a citation, procedure, PPE item, or technical fact. Return "
        "only JSON matching the supplied schema. If evidence cannot support a safe action, "
        "return MANUAL_REVIEW_REQUIRED with a concise reason and null mitigation_plan."
    )
    user = json.dumps(
        {
            "task": "Generate a conservative mitigation plan grounded in the evidence.",
            "predicted_failure": predicted_failure,
            "confidence_score": confidence_score,
            "allowed_evidence": _evidence_payload(evidence),
        },
        ensure_ascii=False,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class MitigationGenerator:
    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = 30.0,
        max_attempts: int = 2,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.client_factory = client_factory

    def generate(
        self,
        predicted_failure: str,
        confidence_score: float,
        evidence: list[dict[str, Any]],
        evidence_is_sufficient: bool,
    ) -> MitigationResponse:
        if not evidence_is_sufficient or not evidence:
            return manual_review(
                predicted_failure,
                confidence_score,
                "Insufficient verified evidence for automated mitigation.",
            )

        allowed_citations = {
            (str(item["document_id"]), int(item["page"]), str(item["title"]))
            for item in evidence
        }
        last_error = "Local model did not return a valid response."
        for _ in range(self.max_attempts):
            try:
                response = self._request(predicted_failure, confidence_score, evidence)
                parsed = MitigationResponse.model_validate_json(response)
                if parsed.predicted_failure != predicted_failure:
                    raise ValueError("model changed the predicted failure")
                if abs(parsed.confidence_score - confidence_score) > 1e-6:
                    raise ValueError("model changed the confidence score")
                if parsed.status == "MITIGATION_READY":
                    assert parsed.mitigation_plan is not None
                    citations = {
                        (item.document_id, item.page, item.title)
                        for item in parsed.mitigation_plan.citations
                    }
                    if not citations.issubset(allowed_citations):
                        raise ValueError("model produced a citation outside supplied evidence")
                return parsed
            except (httpx.HTTPError, KeyError, TypeError, ValueError, ValidationError) as exc:
                last_error = f"Validated local generation failed: {type(exc).__name__}."

        return manual_review(predicted_failure, confidence_score, last_error)

    def _request(
        self,
        predicted_failure: str,
        confidence_score: float,
        evidence: list[dict[str, Any]],
    ) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "format": MitigationResponse.model_json_schema(),
            "messages": _messages(predicted_failure, confidence_score, evidence),
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 350},
        }
        with self.client_factory(timeout=self.timeout_seconds) as client:
            result = client.post(f"{self.base_url}/api/chat", json=payload)
            result.raise_for_status()
            return str(result.json()["message"]["content"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure", default="OSF")
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--evidence-json", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_OLLAMA_URL)
    args = parser.parse_args()
    evidence = json.loads(args.evidence_json)
    result = MitigationGenerator(base_url=args.base_url, model=args.model).generate(
        args.failure, args.confidence, evidence, evidence_is_sufficient=bool(evidence)
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
