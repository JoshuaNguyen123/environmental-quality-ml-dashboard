"""Tests for descriptive statistics — locks in bias-corrected estimators."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.stats.descriptive import compute_descriptive_stats


def test_moment_keys_match_bias_corrected_contract():
    s = pd.Series(np.arange(1.0, 11.0))
    out = compute_descriptive_stats(pd.DataFrame({"x": s}), ["x"])
    stats = out["column_statistics"]["x"]
    # Variance/std use ddof=1 (sample); skew/kurtosis use bias=False.
    assert math.isclose(stats["variance"], float(s.var()), rel_tol=1e-12)
    assert math.isclose(stats["std"], float(s.std()), rel_tol=1e-12)
    assert math.isclose(stats["skewness"], float(sp_stats.skew(s, bias=False)), rel_tol=1e-12)
    assert math.isclose(
        stats["excess_kurtosis"],
        float(sp_stats.kurtosis(s, bias=False)),
        rel_tol=1e-12,
    )


def test_excess_kurtosis_key_present_not_kurtosis():
    s = np.random.default_rng(0).normal(size=200)
    out = compute_descriptive_stats(pd.DataFrame({"x": s}), ["x"])
    stats = out["column_statistics"]["x"]
    assert "excess_kurtosis" in stats
    assert "kurtosis" not in stats
    # Excess kurtosis of a normal sample should be near zero.
    assert abs(stats["excess_kurtosis"]) < 1.0


def test_correlation_matrix_diagonal_is_one():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
    out = compute_descriptive_stats(df, ["a", "b"])
    corr = out["correlation_matrix"]
    assert math.isclose(corr["a"]["a"], 1.0, abs_tol=1e-12)
    assert math.isclose(corr["b"]["b"], 1.0, abs_tol=1e-12)
