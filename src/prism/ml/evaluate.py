"""Evaluation helpers and CLI for a saved PRISM baseline model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from prism.config import load_training_config
from prism.ml.artifacts import MODEL_FILENAME, load_bundle
from prism.ml.features import load_dataset, prepare_training_data


def evaluate_pipeline(
    pipeline: Any, features: Any, target: Any, threshold: float
) -> dict[str, Any]:
    probabilities = pipeline.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(target, predictions).tolist(),
        "classification_report": classification_report(
            target, predictions, output_dict=True, zero_division=0
        ),
        "probability_summary": {
            "minimum": float(np.min(probabilities)),
            "maximum": float(np.max(probabilities)),
            "mean": float(np.mean(probabilities)),
        },
    }


def evaluate_saved_model(config_path: str | Path, artifact_path: str | Path) -> dict[str, Any]:
    config = load_training_config(config_path)
    frame = load_dataset(config.data_path)
    features, target = prepare_training_data(frame, config.target_column)
    _, test_features, _, test_target = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=target,
    )
    bundle = load_bundle(artifact_path)
    threshold = float(bundle["metadata"]["decision_threshold"])
    return evaluate_pipeline(bundle["pipeline"], test_features, test_target, threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.json")
    parser.add_argument("--artifact", default=f"artifacts/{MODEL_FILENAME}")
    args = parser.parse_args()
    print(json.dumps(evaluate_saved_model(args.config, args.artifact), indent=2))


if __name__ == "__main__":
    main()
