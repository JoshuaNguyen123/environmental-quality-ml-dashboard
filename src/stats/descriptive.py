"""Descriptive statistics: moments, correlations, and distribution summaries."""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from typing import Dict, Any


def compute_descriptive_stats(df: pd.DataFrame, numeric_cols: list | None = None) -> Dict[str, Any]:
    """Compute descriptive statistics for all numeric columns.

    Returns dict with per-column: mean, variance, std, skewness, kurtosis,
    min, max, median, and the full Pearson correlation matrix.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    col_stats = {}
    for col in numeric_cols:
        s = df[col].dropna()
        col_stats[col] = {
            "mean": float(s.mean()),
            "variance": float(s.var()),
            "std": float(s.std()),
            "skewness": float(sp_stats.skew(s)),
            "kurtosis": float(sp_stats.kurtosis(s)),
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "n": int(len(s)),
        }

    corr = df[numeric_cols].corr()

    return {
        "column_statistics": col_stats,
        "correlation_matrix": corr.to_dict(),
    }
