"""Feature engineering: temporal features, pollution target, and lag features."""

import numpy as np
import pandas as pd


def engineer_features(
    df: pd.DataFrame,
    regulatory_threshold: float = 200,
    datetime_col: str = "DateTime",
) -> pd.DataFrame:
    """Create model-ready features.

    Adds:
        - Hour, DayOfWeek, Month, Season, IsWeekend
        - Hour cyclical encoding (sin/cos)
        - High_Pollution binary target
        - Lag features (1 h, 3 h, 6 h) for NO2
        - Rolling means (3 h, 6 h) for NO2
    """
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col])

    # -- Temporal features ------------------------------------------------
    df["Hour"] = dt.dt.hour
    df["DayOfWeek"] = dt.dt.dayofweek
    df["Month"] = dt.dt.month
    df["IsWeekend"] = (dt.dt.dayofweek >= 5).astype(int)

    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                  6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    df["Season"] = df["Month"].map(season_map)

    # Cyclical hour encoding
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    # -- Classification target --------------------------------------------
    df["High_Pollution"] = (df["NO2_GT"] > regulatory_threshold).astype(int)

    # -- Lag features -----------------------------------------------------
    for lag in [1, 3, 6]:
        df[f"NO2_lag_{lag}h"] = df["NO2_GT"].shift(lag)

    # -- Rolling statistics (shifted by 1 to prevent target leakage) ----
    for window in [3, 6]:
        df[f"NO2_roll_mean_{window}h"] = (
            df["NO2_GT"].rolling(window=window, min_periods=1).mean().shift(1)
        )

    # Drop rows with NaN from lagging
    df = df.dropna().reset_index(drop=True)

    return df


def get_feature_columns(df: pd.DataFrame, exclude: list | None = None) -> list:
    """Return feature column names, excluding targets and metadata."""
    if exclude is None:
        exclude = ["DateTime", "NO2_GT", "High_Pollution"]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in exclude]
