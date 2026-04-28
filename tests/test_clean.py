"""Tests for the data cleaning layer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.clean import clean


def _make_raw(n: int = 24) -> pd.DataFrame:
    """Build a minimal raw frame in UCI shape (Date + Time + numerics)."""
    times = pd.date_range("2004-03-10 00:00", periods=n, freq="h")
    return pd.DataFrame({
        "Date": times.strftime("%d/%m/%Y"),
        "Time": times.strftime("%H.%M.%S"),
        "CO_GT": np.linspace(1.0, 2.0, n),
        "NO2_GT": np.linspace(50, 150, n),
        "Temperature": np.linspace(10, 18, n),
    })


def test_clean_replaces_sentinel_with_nan_then_fills():
    df = _make_raw()
    df.loc[3, "CO_GT"] = -200
    out = clean(df)
    assert "DateTime" in out.columns
    # The sentinel hole should have been interpolated, so no NaN survives.
    assert out["CO_GT"].isna().sum() == 0
    # Interpolated value should sit between its neighbours.
    interpolated = out["CO_GT"].iloc[3]
    assert 1.0 < interpolated < 2.0


def test_clean_drops_columns_over_50pct_missing():
    df = _make_raw(n=20)
    df["NMHC_GT"] = -200  # 100% sentinel
    out = clean(df)
    assert "NMHC_GT" not in out.columns


def test_clean_sorts_chronologically():
    df = _make_raw(n=12)
    # Reverse the order to test that clean() restores chronology.
    df = df.iloc[::-1].reset_index(drop=True)
    out = clean(df)
    assert out["DateTime"].is_monotonic_increasing


def test_clean_parses_date_time():
    df = _make_raw(n=5)
    out = clean(df)
    assert pd.api.types.is_datetime64_any_dtype(out["DateTime"])
    assert "Date" not in out.columns
    assert "Time" not in out.columns
