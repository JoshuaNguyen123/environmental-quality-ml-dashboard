"""Data cleaning: handle missing values, parse datetimes, standardize columns."""

import numpy as np
import pandas as pd


def clean(df: pd.DataFrame, missing_sentinel: int = -200) -> pd.DataFrame:
    """Clean raw air quality data.

    Steps:
        1. Parse Date + Time -> DateTime index
        2. Replace sentinel missing values (-200) with NaN
        3. Drop columns with >50 % missing (NMHC_GT)
        4. Interpolate remaining gaps (linear, time-aware)
        5. Drop any residual NaN rows
        6. Sort chronologically
    """
    df = df.copy()

    # -- 1. DateTime ------------------------------------------------------
    if "Date" in df.columns and "Time" in df.columns:
        df["DateTime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"].str.replace(".", ":", regex=False),
            dayfirst=True,
            errors="coerce",
        )
        df = df.drop(columns=["Date", "Time"])
    elif "DateTime" not in df.columns:
        raise ValueError("Cannot find Date/Time or DateTime columns")

    df = df.dropna(subset=["DateTime"])
    df = df.sort_values("DateTime").reset_index(drop=True)

    # -- 2. Replace sentinel ----------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace(missing_sentinel, np.nan)

    # -- 3. Drop columns with >50 % missing ------------------------------
    frac_missing = df[numeric_cols].isna().mean()
    drop_cols = frac_missing[frac_missing > 0.5].index.tolist()
    if drop_cols:
        print(f"[clean] Dropping high-missing columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # -- 4. Interpolation ------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=5)

    # -- 5. Drop residual NaN --------------------------------------------
    before = len(df)
    df = df.dropna()
    after = len(df)
    if before - after > 0:
        print(f"[clean] Dropped {before - after} rows with residual NaN")

    df = df.reset_index(drop=True)
    return df
