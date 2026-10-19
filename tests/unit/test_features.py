import pandas as pd
import pytest

from prism.ml.features import (
    FEATURE_COLUMNS,
    engineer_features,
    payload_to_frame,
    prepare_training_data,
)


def sample_frame():
    return pd.DataFrame(
        [
            {
                "Type": "M",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 100,
                "Machine failure": 0,
            }
        ]
    )


def test_engineer_features_does_not_mutate_input():
    source = sample_frame()
    transformed = engineer_features(source)

    assert "temp_diff" not in source.columns
    assert transformed.loc[0, "temp_diff"] == 10.0
    assert transformed.loc[0, "power"] == 60_000.0


def test_prepare_training_data_uses_canonical_feature_order():
    features, target = prepare_training_data(sample_frame())

    assert features.columns.tolist() == FEATURE_COLUMNS
    assert target.tolist() == [0]


def test_payload_to_frame_defaults_to_expected_columns():
    result = payload_to_frame(
        {
            "air_temperature_k": 298.1,
            "process_temperature_k": 308.6,
            "rotational_speed_rpm": 1551,
            "torque_nm": 42.8,
            "tool_wear_min": 21,
        }
    )

    assert result.columns.tolist() == FEATURE_COLUMNS
    assert result.loc[0, "Type"] == "M"
    assert result.loc[0, "temp_diff"] == pytest.approx(10.5)


def test_payload_to_frame_rejects_missing_metrics():
    with pytest.raises(ValueError, match="missing required metrics"):
        payload_to_frame({"air_temperature_k": 298.1})
