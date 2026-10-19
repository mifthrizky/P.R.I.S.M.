"""Run binary failure inference from a versioned PRISM model bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prism.ml.artifacts import MODEL_FILENAME, load_bundle
from prism.ml.features import payload_to_frame

DEMO_PAYLOADS = [
    {
        "sensor_id": "CNC-Milling-01",
        "timestamp": "2026-08-31T08:00:00Z",
        "metrics": {
            "air_temperature_k": 298.1,
            "process_temperature_k": 308.6,
            "rotational_speed_rpm": 1551,
            "torque_nm": 42.8,
            "tool_wear_min": 21,
        },
    },
    {
        "sensor_id": "CNC-Milling-01",
        "timestamp": "2026-08-31T08:15:00Z",
        "metrics": {
            "air_temperature_k": 304.5,
            "process_temperature_k": 313.2,
            "rotational_speed_rpm": 1201,
            "torque_nm": 71.2,
            "tool_wear_min": 215,
        },
    },
]


def predict_payload(bundle: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    machine_type_was_supplied = "machine_type" in payload
    machine_type = str(payload.get("machine_type", "M"))
    features = payload_to_frame(payload["metrics"], machine_type)
    probability = float(bundle["pipeline"].predict_proba(features)[0, 1])
    threshold = float(bundle["metadata"]["decision_threshold"])
    return {
        "sensor_id": payload.get("sensor_id"),
        "timestamp": payload.get("timestamp"),
        "status": "CRITICAL_WARNING" if probability >= threshold else "HEALTHY",
        "failure_probability": probability,
        "threshold": threshold,
        "machine_type": machine_type,
        "machine_type_assumed": not machine_type_was_supplied,
        "model_version": bundle["metadata"]["model_version"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=f"artifacts/{MODEL_FILENAME}")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Path to a sensor payload JSON file")
    source.add_argument("--demo", action="store_true", help="Run the two PRD payloads")
    args = parser.parse_args()

    bundle = load_bundle(args.artifact)
    if args.demo:
        payloads = DEMO_PAYLOADS
    else:
        with args.input.open(encoding="utf-8") as input_file:
            payloads = [json.load(input_file)]
    results = [predict_payload(bundle, payload) for payload in payloads]
    print(json.dumps(results if args.demo else results[0], indent=2))


if __name__ == "__main__":
    main()
