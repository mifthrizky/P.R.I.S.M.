"""Configuration loading for reproducible PRISM training runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    data_path: Path
    artifact_dir: Path
    target_column: str
    test_size: float
    validation_size: float
    random_seed: int
    minimum_failure_recall: float
    minimum_failure_type_recall: float
    cross_validation_folds: int
    failure_type_columns: tuple[str, ...]
    model_candidates: dict[str, dict[str, Any]]
    failure_classifier: dict[str, Any]


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load and validate a training configuration from JSON."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as config_file:
        raw: dict[str, Any] = json.load(config_file)

    test_size = float(raw["test_size"])
    validation_size = float(raw["validation_size"])
    minimum_recall = float(raw["minimum_failure_recall"])
    minimum_type_recall = float(raw["minimum_failure_type_recall"])
    folds = int(raw["cross_validation_folds"])
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < validation_size < 1 or test_size + validation_size >= 1:
        raise ValueError("validation_size must leave room for training data")
    if not 0 < minimum_recall <= 1:
        raise ValueError("minimum_failure_recall must be between 0 and 1")
    if not 0 < minimum_type_recall <= 1:
        raise ValueError("minimum_failure_type_recall must be between 0 and 1")
    if folds < 2:
        raise ValueError("cross_validation_folds must be at least 2")
    candidates = dict(raw["model_candidates"])
    required_candidates = {"random_forest", "xgboost"}
    if set(candidates) != required_candidates:
        raise ValueError(f"model_candidates must contain exactly {sorted(required_candidates)}")

    base_dir = config_path.parent
    return TrainingConfig(
        data_path=(base_dir / raw["data_path"]).resolve(),
        artifact_dir=(base_dir / raw["artifact_dir"]).resolve(),
        target_column=str(raw["target_column"]),
        test_size=test_size,
        validation_size=validation_size,
        random_seed=int(raw["random_seed"]),
        minimum_failure_recall=minimum_recall,
        minimum_failure_type_recall=minimum_type_recall,
        cross_validation_folds=folds,
        failure_type_columns=tuple(raw["failure_type_columns"]),
        model_candidates=candidates,
        failure_classifier=dict(raw["failure_classifier"]),
    )
