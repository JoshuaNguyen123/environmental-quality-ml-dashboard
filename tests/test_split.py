"""Tests for chronological splitting."""

from __future__ import annotations

import pandas as pd

from src.data.split import chronological_split


def _make_df(n: int = 100) -> pd.DataFrame:
    times = pd.date_range("2004-03-10", periods=n, freq="h")
    return pd.DataFrame({"DateTime": times, "value": range(n)})


def test_split_ratios_default():
    df = _make_df(100)
    train, val, test = chronological_split(df)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15


def test_split_preserves_temporal_order():
    df = _make_df(100)
    train, val, test = chronological_split(df)
    assert train["DateTime"].max() < val["DateTime"].min()
    assert val["DateTime"].max() < test["DateTime"].min()


def test_split_does_not_lose_rows():
    df = _make_df(100)
    train, val, test = chronological_split(df)
    assert len(train) + len(val) + len(test) == len(df)


def test_split_custom_ratios():
    df = _make_df(100)
    train, val, test = chronological_split(df, train_ratio=0.6, val_ratio=0.2)
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20
