"""Versioned persistence for PRISM model bundles and metadata."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib

ARTIFACT_FORMAT_VERSION = 1
MODEL_FILENAME = "model_bundle.joblib"
METADATA_FILENAME = "model_metadata.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "joblib": version("joblib"),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scikit-learn": version("scikit-learn"),
        "xgboost": version("xgboost"),
    }


def save_bundle(
    pipeline: Any,
    metadata: dict[str, Any],
    artifact_dir: str | Path,
    failure_type_pipeline: Any | None = None,
) -> Path:
    """Atomically write a model bundle and its human-readable metadata."""
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / MODEL_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    bundle_temp = output_dir / f"{MODEL_FILENAME}.tmp"
    metadata_temp = output_dir / f"{METADATA_FILENAME}.tmp"

    bundle = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "pipeline": pipeline,
        "failure_type_pipeline": failure_type_pipeline,
        "metadata": metadata,
    }
    joblib.dump(bundle, bundle_temp, compress=3)
    with metadata_temp.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(metadata, target, indent=2, sort_keys=True)
        target.write("\n")
    bundle_temp.replace(bundle_path)
    metadata_temp.replace(metadata_path)
    return bundle_path


def load_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("Unsupported legacy artifact: expected a versioned bundle")
    if bundle.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError("Unsupported model artifact format version")
    if "pipeline" not in bundle or "metadata" not in bundle:
        raise ValueError("Model bundle is missing pipeline or metadata")
    return bundle
