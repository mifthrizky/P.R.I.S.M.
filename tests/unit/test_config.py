import json

import pytest

from prism.config import load_training_config


def test_load_training_config_resolves_relative_paths(tmp_path):
    config_path = tmp_path / "training.json"
    config_path.write_text(
        json.dumps(
            {
                "data_path": "data.csv",
                "artifact_dir": "artifacts",
                "target_column": "Machine failure",
                "test_size": 0.2,
                "validation_size": 0.2,
                "random_seed": 42,
                "minimum_failure_recall": 0.9,
                "minimum_failure_type_recall": 0.8,
                "cross_validation_folds": 5,
                "failure_type_columns": ["TWF", "HDF", "PWF", "OSF"],
                "model_candidates": {
                    "random_forest": {"n_estimators": 10},
                    "xgboost": {"n_estimators": 10},
                },
                "failure_classifier": {"n_estimators": 10},
            }
        ),
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config.data_path == (tmp_path / "data.csv").resolve()
    assert config.artifact_dir == (tmp_path / "artifacts").resolve()
    assert config.minimum_failure_recall == 0.9


@pytest.mark.parametrize(
    ("field", "value"),
    [("test_size", 0), ("test_size", 1), ("minimum_failure_recall", -0.1)],
)
def test_load_training_config_rejects_invalid_ranges(tmp_path, field, value):
    raw = {
        "data_path": "data.csv",
        "artifact_dir": "artifacts",
        "target_column": "Machine failure",
        "test_size": 0.2,
        "validation_size": 0.2,
        "random_seed": 42,
        "minimum_failure_recall": 0.9,
        "minimum_failure_type_recall": 0.8,
        "cross_validation_folds": 5,
        "failure_type_columns": ["TWF", "HDF", "PWF", "OSF"],
        "model_candidates": {
            "random_forest": {"n_estimators": 10},
            "xgboost": {"n_estimators": 10},
        },
        "failure_classifier": {"n_estimators": 10},
    }
    raw[field] = value
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError):
        load_training_config(config_path)
