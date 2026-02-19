"""Preprocessing pipelines: scaling and encoding for ML models."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from typing import Tuple


def build_preprocessor(feature_cols: list) -> ColumnTransformer:
    """Build a StandardScaler for all numeric features."""
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), feature_cols),
        ],
        remainder="drop",
    )


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract X matrix and y vector from a DataFrame."""
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)
    return X, y
