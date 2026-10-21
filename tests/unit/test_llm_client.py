import json

import httpx
import pytest
from pydantic import ValidationError

from prism.llm.client import MitigationGenerator, retrieval_query
from prism.llm.schemas import MitigationResponse

EVIDENCE = [
    {
        "document_id": "OSHA-3120",
        "title": "Control of Hazardous Energy",
        "page": 7,
        "text": "Isolate the machine from its energy source and apply lockout devices.",
    }
]


def ollama_response(content: dict) -> httpx.Response:
    request = httpx.Request("POST", "http://ollama/api/chat")
    return httpx.Response(
        200,
        request=request,
        json={"message": {"role": "assistant", "content": json.dumps(content)}},
    )


def valid_payload() -> dict:
    return {
        "status": "MITIGATION_READY",
        "predicted_failure": "OSF",
        "confidence_score": 0.9,
        "mitigation_plan": {
            "immediate_action": "Stop the machine and isolate hazardous energy.",
            "required_ppe": ["Safety glasses"],
            "safety_steps": ["Apply lockout devices before maintenance."],
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


def generator_for(handler, max_attempts: int = 2) -> MitigationGenerator:
    transport = httpx.MockTransport(handler)
    return MitigationGenerator(
        base_url="http://ollama",
        max_attempts=max_attempts,
        client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs),
    )


def test_valid_grounded_response_is_accepted() -> None:
    generator = generator_for(lambda request: ollama_response(valid_payload()))
    result = generator.generate("OSF", 0.9, EVIDENCE, evidence_is_sufficient=True)
    assert result.status == "MITIGATION_READY"
    assert result.mitigation_plan is not None
    assert result.mitigation_plan.citations[0].page == 7


def test_fabricated_citation_retries_then_falls_back() -> None:
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        payload = valid_payload()
        payload["mitigation_plan"]["citations"][0]["page"] = 99
        return ollama_response(payload)

    result = generator_for(handler).generate("OSF", 0.9, EVIDENCE, True)
    assert attempts == 2
    assert result.status == "MANUAL_REVIEW_REQUIRED"
    assert result.mitigation_plan is None


def test_invalid_json_falls_back_without_leaking_raw_output() -> None:
    def handler(request):
        return httpx.Response(
            200,
            request=httpx.Request("POST", "http://ollama/api/chat"),
            json={"message": {"content": "not-json"}},
        )

    result = generator_for(handler).generate("PWF", 0.81, EVIDENCE, True)
    assert result.status == "MANUAL_REVIEW_REQUIRED"
    assert "not-json" not in result.reason


def test_insufficient_evidence_never_calls_model() -> None:
    def handler(request):
        raise AssertionError("model must not be called")

    result = generator_for(handler).generate("HDF", 0.7, [], False)
    assert result.status == "MANUAL_REVIEW_REQUIRED"
    assert "Insufficient" in result.reason


def test_schema_rejects_extra_fields() -> None:
    payload = valid_payload()
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        MitigationResponse.model_validate(payload)


def test_failure_query_is_specific_and_has_safe_default() -> None:
    assert "overstrain" in retrieval_query("OSF")
    assert "lockout" in retrieval_query("UNKNOWN")
