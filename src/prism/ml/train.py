"""Train and persist the reproducible PRISM binary failure baseline."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prism import __version__
from prism.config import TrainingConfig, load_training_config
from prism.ml.artifacts import runtime_versions, save_bundle, sha256_file
from prism.ml.evaluate import evaluate_pipeline
from prism.ml.features import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    load_dataset,
    prepare_training_data,
)


def build_pipeline(config: TrainingConfig) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURE_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURE_COLUMNS,
            ),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=config.model.n_estimators,
        class_weight=config.model.class_weight,
        max_depth=config.model.max_depth,
        random_state=config.random_seed,
        n_jobs=config.model.n_jobs,
    )
    return Pipeline([("prep", preprocessor), ("clf", classifier)])


def train(config: TrainingConfig) -> tuple[Pipeline, dict[str, object]]:
    frame = load_dataset(config.data_path)
    features, target = prepare_training_data(frame, config.target_column)
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=target,
    )

    pipeline = build_pipeline(config)
    pipeline.fit(train_features, train_target)
    holdout_metrics = evaluate_pipeline(
        pipeline, test_features, test_target, config.decision_threshold
    )

    cross_validation = StratifiedKFold(
        n_splits=config.cross_validation_folds,
        shuffle=True,
        random_state=config.random_seed,
    )
    cv_scores = cross_val_score(
        build_pipeline(config),
        features,
        target,
        cv=cross_validation,
        scoring="roc_auc",
    )

    metadata: dict[str, object] = {
        "project_version": __version__,
        "model_version": "1.0.0",
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "model_type": "RandomForestClassifier",
        "target_column": config.target_column,
        "feature_columns": FEATURE_COLUMNS,
        "decision_threshold": config.decision_threshold,
        "random_seed": config.random_seed,
        "dataset": {
            "filename": config.data_path.name,
            "sha256": sha256_file(config.data_path),
            "rows": int(len(frame)),
            "positive_rows": int(target.sum()),
        },
        "split": {
            "test_size": config.test_size,
            "train_rows": int(len(train_features)),
            "test_rows": int(len(test_features)),
        },
        "model_parameters": {
            "n_estimators": config.model.n_estimators,
            "class_weight": config.model.class_weight,
            "max_depth": config.model.max_depth,
            "n_jobs": config.model.n_jobs,
        },
        "metrics": {
            "holdout": holdout_metrics,
            "cross_validation": {
                "folds": config.cross_validation_folds,
                "roc_auc_scores": [float(score) for score in cv_scores],
                "roc_auc_mean": float(cv_scores.mean()),
                "roc_auc_std": float(cv_scores.std()),
            },
        },
        "runtime_versions": runtime_versions(),
    }
    return pipeline, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.json")
    args = parser.parse_args()
    config = load_training_config(args.config)
    pipeline, metadata = train(config)
    bundle_path = save_bundle(pipeline, metadata, config.artifact_dir)
    summary = {
        "artifact": str(bundle_path),
        "holdout_roc_auc": metadata["metrics"]["holdout"]["roc_auc"],
        "cross_validation_roc_auc": metadata["metrics"]["cross_validation"][
            "roc_auc_mean"
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
