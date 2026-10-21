import json

import httpx

from prism.llm.client import MitigationGenerator

EVIDENCE = [
    {
        "document_id": "OSHA-3120",
        "title": "Control of Hazardous Energy",
        "page": 7,
        "text": "Isolate energy and apply lockout devices.",
    }
]


def valid_response() -> dict:
    return {
        "status": "MITIGATION_READY",
        "predicted_failure": "OSF",
        "confidence_score": 0.9,
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


def test_one_hundred_responses_pass_schema_and_grounding() -> None:
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            request=request,
            json={"message": {"content": json.dumps(valid_response())}},
        )

    transport = httpx.MockTransport(handler)
    generator = MitigationGenerator(
        base_url="http://ollama",
        client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs),
    )
    results = [generator.generate("OSF", 0.9, EVIDENCE, True) for _ in range(100)]
    assert calls == 100
    assert all(result.status == "MITIGATION_READY" for result in results)


def test_timeout_fault_is_contained_by_safe_fallback() -> None:
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("injected timeout", request=request)

    transport = httpx.MockTransport(handler)
    generator = MitigationGenerator(
        base_url="http://ollama",
        client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs),
    )
    result = generator.generate("OSF", 0.9, EVIDENCE, True)
    assert calls == 2
    assert result.status == "MANUAL_REVIEW_REQUIRED"


def test_server_error_fault_is_contained_by_safe_fallback() -> None:
    def handler(request):
        return httpx.Response(503, request=request, json={"error": "unavailable"})

    transport = httpx.MockTransport(handler)
    generator = MitigationGenerator(
        base_url="http://ollama",
        client_factory=lambda **kwargs: httpx.Client(transport=transport, **kwargs),
    )
    result = generator.generate("PWF", 0.8, EVIDENCE, True)
    assert result.status == "MANUAL_REVIEW_REQUIRED"
