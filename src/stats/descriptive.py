"""Descriptive statistics: moments, correlations, and distribution summaries."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def compute_descriptive_stats(df: pd.DataFrame, numeric_cols: list | None = None) -> dict[str, Any]:
    """Compute descriptive statistics for all numeric columns.

    Per-column statistics use bias-corrected (sample) estimators consistently:
    variance/std with ``ddof=1``, skewness and excess kurtosis with
    ``bias=False``. ``excess_kurtosis`` is Fisher's definition (raw kurtosis
    minus 3); a normal distribution gives 0. The Pearson correlation matrix
    uses pairwise complete observations.
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
            "skewness": float(sp_stats.skew(s, bias=False)),
            "excess_kurtosis": float(sp_stats.kurtosis(s, bias=False)),
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
