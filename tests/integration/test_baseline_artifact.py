from pathlib import Path

import pytest

from prism.ml.artifacts import load_bundle, sha256_file
from prism.ml.evaluate import evaluate_saved_model
from prism.ml.inference import DEMO_PAYLOADS, predict_payload

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training.json"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "model_bundle.joblib"
DATASET_PATH = PROJECT_ROOT / "ai4i2020.csv"


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(ARTIFACT_PATH)


def test_artifact_matches_dataset_and_feature_contract(bundle):
    metadata = bundle["metadata"]

    assert metadata["dataset"]["sha256"] == sha256_file(DATASET_PATH)
    assert metadata["dataset"]["rows"] == 10_000
    assert 0 < metadata["decision_threshold"] < 1
    assert metadata["model_version"] == "2.0.0"
    assert bundle["failure_type_pipeline"] is not None
    assert set(metadata["failure_type_columns"]) == {"TWF", "HDF", "PWF", "OSF"}


def test_saved_model_meets_roc_auc_gate():
    metrics = evaluate_saved_model(CONFIG_PATH, ARTIFACT_PATH)

    assert metrics["roc_auc"] >= 0.88
    assert metrics["classification_report"]["1"]["recall"] >= 0.90


def test_prd_demo_payloads_take_expected_branches(bundle):
    normal = predict_payload(bundle, DEMO_PAYLOADS[0])
    anomaly = predict_payload(bundle, DEMO_PAYLOADS[1])

    assert normal["status"] == "HEALTHY"
    assert normal["failure_probability"] < normal["threshold"]
    assert anomaly["status"] == "CRITICAL_WARNING"
    assert anomaly["failure_probability"] >= anomaly["threshold"]
    assert anomaly["primary_failure"] == "OSF"
    assert "OSF" in {
        prediction["failure_type"] for prediction in anomaly["failure_predictions"]
    }
    assert normal["machine_type"] == "M"
    assert normal["machine_type_assumed"] is True
