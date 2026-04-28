"""Tests for the data ingestion layer."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.ingest import (
    REQUIRED_COLUMNS,
    SchemaError,
    generate_synthetic_air_quality,
    load_or_generate,
    validate_schema,
)


def test_synthetic_generator_default_size():
    df = generate_synthetic_air_quality()
    assert len(df) == 9357
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"missing column: {col}"


def test_synthetic_generator_seeded_reproducible():
    df1 = generate_synthetic_air_quality(n_hours=200, seed=7)
    df2 = generate_synthetic_air_quality(n_hours=200, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_generator_small():
    df = generate_synthetic_air_quality(n_hours=120)
    assert len(df) == 120
    # ~3% of NO2 values are sentinel; for n=120 we expect at least 1 valid row.
    valid_no2 = (df["NO2_GT"] != -200).sum()
    assert valid_no2 > 50


def test_validate_schema_passes_on_synthetic():
    df = generate_synthetic_air_quality(n_hours=10)
    validate_schema(df)  # should not raise


def test_validate_schema_raises_on_missing_columns():
    df = pd.DataFrame({"Date": ["01/01/2004"], "Time": ["00.00.00"]})
    with pytest.raises(SchemaError) as excinfo:
        validate_schema(df, source="bogus.csv")
    assert "missing required columns" in str(excinfo.value)
    assert "bogus.csv" in str(excinfo.value)


def test_load_or_generate_falls_back_to_synthetic(tmp_path):
    raw_path = tmp_path / "raw" / "air_quality.csv"
    df = load_or_generate(raw_path)
    assert raw_path.exists(), "synthetic CSV should be persisted to disk"
    assert len(df) == 9357


def test_load_or_generate_round_trip(tmp_path):
    raw_path = tmp_path / "raw" / "air_quality.csv"
    df_first = load_or_generate(raw_path)
    df_second = load_or_generate(raw_path)
    # Second call reads the file we wrote on the first call; row counts must match.
    assert len(df_first) == len(df_second)
    for col in REQUIRED_COLUMNS:
        assert col in df_second.columns
