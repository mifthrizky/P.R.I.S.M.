"""Canonical feature definitions and transformations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

RAW_FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

NUMERIC_FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_diff",
    "power",
]

CATEGORICAL_FEATURE_COLUMNS = ["Type"]
FEATURE_COLUMNS = ["Type", *NUMERIC_FEATURE_COLUMNS]

PAYLOAD_TO_DATASET_COLUMNS = {
    "air_temperature_k": "Air temperature [K]",
    "process_temperature_k": "Process temperature [K]",
    "rotational_speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load AI4I data and fail early when required columns are absent."""
    frame = pd.read_csv(path)
    required = {*RAW_FEATURE_COLUMNS, "Machine failure"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return frame


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with deterministic thermal and power features."""
    transformed = frame.copy()
    transformed["temp_diff"] = (
        transformed["Process temperature [K]"]
        - transformed["Air temperature [K]"]
    )
    transformed["power"] = (
        transformed["Torque [Nm]"]
        * transformed["Rotational speed [rpm]"]
    )
    return transformed


def prepare_training_data(
    frame: pd.DataFrame, target_column: str = "Machine failure"
) -> tuple[pd.DataFrame, pd.Series]:
    """Create the canonical feature matrix and binary target."""
    if target_column not in frame.columns:
        raise ValueError(f"Dataset is missing target column: {target_column}")
    transformed = engineer_features(frame)
    return transformed[FEATURE_COLUMNS], transformed[target_column].astype(int)


def payload_to_frame(
    metrics: Mapping[str, Any], machine_type: str = "M"
) -> pd.DataFrame:
    """Convert API-style metric keys into the model's feature contract."""
    if machine_type not in {"L", "M", "H"}:
        raise ValueError("machine_type must be one of L, M, or H")
    missing = sorted(set(PAYLOAD_TO_DATASET_COLUMNS).difference(metrics))
    if missing:
        raise ValueError(f"Payload is missing required metrics: {missing}")

    row: dict[str, Any] = {"Type": machine_type}
    for payload_key, dataset_column in PAYLOAD_TO_DATASET_COLUMNS.items():
        row[dataset_column] = float(metrics[payload_key])
    return engineer_features(pd.DataFrame([row]))[FEATURE_COLUMNS]
