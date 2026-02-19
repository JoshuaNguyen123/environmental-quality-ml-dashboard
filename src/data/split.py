"""Train / Validation / Test splitting -- chronological for time-series integrity."""

import pandas as pd
from typing import Tuple


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    datetime_col: str = "DateTime",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically (no shuffling) to prevent data leakage.

    Returns:
        (train, val, test) DataFrames
    """
    df = df.sort_values(datetime_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    print(f"[split] Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test
