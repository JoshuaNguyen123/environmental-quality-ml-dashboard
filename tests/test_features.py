"""Tests for feature engineering — focuses on the no-leakage invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineer import engineer_features, get_feature_columns


def _make_clean(n: int = 48) -> pd.DataFrame:
    times = pd.date_range("2004-03-10 00:00", periods=n, freq="h")
    return pd.DataFrame({
        "DateTime": times,
        "NO2_GT": np.arange(1, n + 1, dtype=float),
        "Temperature": np.linspace(5, 25, n),
        "Rel_Humidity": np.linspace(40, 80, n),
    })


def test_temporal_features_added():
    df = engineer_features(_make_clean())
    for col in ("Hour", "DayOfWeek", "Month", "IsWeekend", "Hour_sin", "Hour_cos"):
        assert col in df.columns


def test_classification_target_uses_threshold():
    df = engineer_features(_make_clean(n=24), regulatory_threshold=10)
    # Row r has NO2_GT == r+1 + lag drops; check target matches NO2_GT > threshold.
    expected = (df["NO2_GT"] > 10).astype(int)
    pd.testing.assert_series_equal(
        df["High_Pollution"], expected, check_names=False
    )


def test_lag_features_have_correct_shift():
    df = engineer_features(_make_clean(n=24))
    # NO2_lag_1h at row r should equal NO2_GT at row r-1 (in the post-drop frame,
    # rows are renumbered; check the relationship element-wise).
    for col, lag in (("NO2_lag_1h", 1), ("NO2_lag_3h", 3), ("NO2_lag_6h", 6)):
        for i in range(lag, len(df)):
            assert df[col].iloc[i] == df["NO2_GT"].iloc[i - lag], (
                f"{col} at row {i} should equal NO2_GT at row {i - lag}"
            )


def test_rolling_means_do_not_leak_target():
    """Critical invariant: the rolling mean at time t must NOT include NO2_GT[t]."""
    df = engineer_features(_make_clean(n=24))
    # At index i, NO2_roll_mean_3h should be the mean of NO2_GT[i-3..i-1] (shifted by 1).
    for i in range(3, len(df)):
        window = df["NO2_GT"].iloc[max(0, i - 3):i].values
        expected = float(np.mean(window))
        assert abs(df["NO2_roll_mean_3h"].iloc[i] - expected) < 1e-9


def test_get_feature_columns_excludes_targets():
    df = engineer_features(_make_clean())
    feats = get_feature_columns(df)
    assert "NO2_GT" not in feats
    assert "High_Pollution" not in feats
    assert "DateTime" not in feats
    # But added features are present.
    assert "Hour_sin" in feats
    assert "NO2_lag_1h" in feats
