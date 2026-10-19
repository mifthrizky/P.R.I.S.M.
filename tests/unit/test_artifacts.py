from prism.ml.artifacts import load_bundle, save_bundle


def test_model_bundle_round_trip(tmp_path):
    pipeline = {"estimator": "test-double"}
    metadata = {"model_version": "test", "decision_threshold": 0.7}

    artifact_path = save_bundle(pipeline, metadata, tmp_path)
    loaded = load_bundle(artifact_path)

    assert loaded["pipeline"] == pipeline
    assert loaded["metadata"] == metadata
    assert (tmp_path / "model_metadata.json").exists()


def test_load_bundle_rejects_legacy_artifact(tmp_path):
    import joblib
    import pytest

    legacy_path = tmp_path / "legacy.joblib"
    joblib.dump(["not", "a", "bundle"], legacy_path)

    with pytest.raises(ValueError, match="legacy artifact"):
        load_bundle(legacy_path)
