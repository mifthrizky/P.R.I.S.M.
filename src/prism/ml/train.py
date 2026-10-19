"""Train calibrated binary and multi-label PRISM predictive models."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from prism import __version__
from prism.config import TrainingConfig, load_training_config
from prism.ml.artifacts import runtime_versions, save_bundle, sha256_file
from prism.ml.evaluate import evaluate_pipeline
from prism.ml.features import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    load_dataset,
    prepare_failure_type_targets,
    prepare_training_data,
)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURE_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURE_COLUMNS,
            ),
        ]
    )


def build_binary_pipeline(
    config: TrainingConfig, model_name: str, target: object
) -> Pipeline:
    parameters = config.model_candidates[model_name]
    if model_name == "random_forest":
        classifier = RandomForestClassifier(
            **parameters,
            random_state=config.random_seed,
        )
    elif model_name == "xgboost":
        positives = float(np.asarray(target).sum())
        negatives = float(len(target) - positives)
        classifier = XGBClassifier(
            **parameters,
            scale_pos_weight=negatives / positives,
            eval_metric="logloss",
            random_state=config.random_seed,
        )
    else:
        raise ValueError(f"Unsupported model candidate: {model_name}")
    return Pipeline([("prep", build_preprocessor()), ("clf", classifier)])


def build_failure_type_pipeline(config: TrainingConfig) -> Pipeline:
    classifier = OneVsRestClassifier(
        RandomForestClassifier(
            **config.failure_classifier,
            random_state=config.random_seed,
        )
    )
    return Pipeline([("prep", build_preprocessor()), ("clf", classifier)])


def select_threshold(target: object, probabilities: np.ndarray, minimum_recall: float) -> float:
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    eligible = [
        (float(candidate_precision), float(candidate_threshold))
        for candidate_precision, candidate_recall, candidate_threshold in zip(
            precision[:-1], recall[:-1], thresholds, strict=True
        )
        if candidate_recall >= minimum_recall
    ]
    if not eligible:
        return 0.0
    return max(eligible, key=lambda item: (item[0], item[1]))[1]


def evaluate_label(
    target: object, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "threshold": float(threshold),
    }


def compare_candidates(
    config: TrainingConfig, features: object, target: object
) -> dict[str, dict[str, object]]:
    cross_validation = StratifiedKFold(
        n_splits=config.cross_validation_folds,
        shuffle=True,
        random_state=config.random_seed,
    )
    results: dict[str, dict[str, object]] = {}
    for model_name in config.model_candidates:
        scores = cross_validate(
            build_binary_pipeline(config, model_name, target),
            features,
            target,
            cv=cross_validation,
            scoring=("roc_auc", "average_precision"),
        )
        results[model_name] = {
            "roc_auc_scores": [float(score) for score in scores["test_roc_auc"]],
            "roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "average_precision_scores": [
                float(score) for score in scores["test_average_precision"]
            ],
            "average_precision_mean": float(scores["test_average_precision"].mean()),
        }
    return results


def train(config: TrainingConfig) -> tuple[Pipeline, Pipeline, dict[str, object]]:
    frame = load_dataset(config.data_path)
    features, target = prepare_training_data(frame, config.target_column)
    failure_targets = prepare_failure_type_targets(frame, config.failure_type_columns)
    train_val_features, test_features, train_val_target, test_target = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=target,
    )
    test_failure_targets = failure_targets.loc[test_features.index]
    relative_validation_size = config.validation_size / (1 - config.test_size)
    train_features, validation_features, train_target, validation_target = train_test_split(
        train_val_features,
        train_val_target,
        test_size=relative_validation_size,
        random_state=config.random_seed,
        stratify=train_val_target,
    )
    train_failure_targets = failure_targets.loc[train_features.index]
    validation_failure_targets = failure_targets.loc[validation_features.index]

    candidate_results = compare_candidates(config, train_val_features, train_val_target)
    selected_model = max(
        candidate_results,
        key=lambda name: (
            candidate_results[name]["average_precision_mean"],
            candidate_results[name]["roc_auc_mean"],
        ),
    )
    calibrated_pipeline = CalibratedClassifierCV(
        build_binary_pipeline(config, selected_model, train_target),
        method="sigmoid",
        cv=3,
    )
    calibrated_pipeline.fit(train_features, train_target)
    validation_probabilities = calibrated_pipeline.predict_proba(validation_features)[:, 1]
    decision_threshold = select_threshold(
        validation_target,
        validation_probabilities,
        config.minimum_failure_recall,
    )
    holdout_metrics = evaluate_pipeline(
        calibrated_pipeline, test_features, test_target, decision_threshold
    )

    failure_type_pipeline = build_failure_type_pipeline(config)
    failure_type_pipeline.fit(train_features, train_failure_targets)
    validation_type_probabilities = failure_type_pipeline.predict_proba(validation_features)
    test_type_probabilities = failure_type_pipeline.predict_proba(test_features)
    failure_type_thresholds: dict[str, float] = {}
    failure_type_metrics: dict[str, dict[str, float]] = {}
    for index, failure_type in enumerate(config.failure_type_columns):
        threshold = select_threshold(
            validation_failure_targets[failure_type],
            validation_type_probabilities[:, index],
            config.minimum_failure_type_recall,
        )
        failure_type_thresholds[failure_type] = threshold
        failure_type_metrics[failure_type] = evaluate_label(
            test_failure_targets[failure_type],
            test_type_probabilities[:, index],
            threshold,
        )

    metadata: dict[str, object] = {
        "project_version": __version__,
        "model_version": "2.0.0",
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "model_type": "CalibratedClassifierCV",
        "selected_binary_model": selected_model,
        "target_column": config.target_column,
        "failure_type_columns": list(config.failure_type_columns),
        "feature_columns": FEATURE_COLUMNS,
        "decision_threshold": decision_threshold,
        "failure_type_thresholds": failure_type_thresholds,
        "random_seed": config.random_seed,
        "dataset": {
            "filename": config.data_path.name,
            "sha256": sha256_file(config.data_path),
            "rows": int(len(frame)),
            "positive_rows": int(target.sum()),
        },
        "split": {
            "test_size": config.test_size,
            "validation_size": config.validation_size,
            "train_rows": int(len(train_features)),
            "validation_rows": int(len(validation_features)),
            "test_rows": int(len(test_features)),
        },
        "model_candidates": config.model_candidates,
        "failure_classifier_parameters": config.failure_classifier,
        "metrics": {
            "holdout": holdout_metrics,
            "candidate_comparison": candidate_results,
            "failure_types": failure_type_metrics,
        },
        "runtime_versions": runtime_versions(),
    }
    return calibrated_pipeline, failure_type_pipeline, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.json")
    args = parser.parse_args()
    config = load_training_config(args.config)
    pipeline, failure_type_pipeline, metadata = train(config)
    bundle_path = save_bundle(
        pipeline,
        metadata,
        config.artifact_dir,
        failure_type_pipeline=failure_type_pipeline,
    )
    summary = {
        "artifact": str(bundle_path),
        "holdout_roc_auc": metadata["metrics"]["holdout"]["roc_auc"],
        "selected_binary_model": metadata["selected_binary_model"],
        "decision_threshold": metadata["decision_threshold"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
