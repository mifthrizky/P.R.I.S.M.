"""Configuration loading for reproducible PRISM training runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    type: str
    n_estimators: int
    class_weight: str
    max_depth: int | None
    n_jobs: int


@dataclass(frozen=True)
class TrainingConfig:
    data_path: Path
    artifact_dir: Path
    target_column: str
    test_size: float
    random_seed: int
    decision_threshold: float
    cross_validation_folds: int
    model: ModelConfig


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load and validate a training configuration from JSON."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as config_file:
        raw: dict[str, Any] = json.load(config_file)

    test_size = float(raw["test_size"])
    threshold = float(raw["decision_threshold"])
    folds = int(raw["cross_validation_folds"])
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 <= threshold <= 1:
        raise ValueError("decision_threshold must be between 0 and 1")
    if folds < 2:
        raise ValueError("cross_validation_folds must be at least 2")

    model_raw = raw["model"]
    model = ModelConfig(
        type=str(model_raw["type"]),
        n_estimators=int(model_raw["n_estimators"]),
        class_weight=str(model_raw["class_weight"]),
        max_depth=model_raw["max_depth"],
        n_jobs=int(model_raw["n_jobs"]),
    )
    if model.type != "random_forest":
        raise ValueError(f"Unsupported baseline model type: {model.type}")

    base_dir = config_path.parent
    return TrainingConfig(
        data_path=(base_dir / raw["data_path"]).resolve(),
        artifact_dir=(base_dir / raw["artifact_dir"]).resolve(),
        target_column=str(raw["target_column"]),
        test_size=test_size,
        random_seed=int(raw["random_seed"]),
        decision_threshold=threshold,
        cross_validation_folds=folds,
        model=model,
    )
