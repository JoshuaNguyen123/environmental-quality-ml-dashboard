"""Hypothesis tests: t-test, ANOVA, Chi-Square with documented assumptions."""

import pandas as pd
from scipy import stats as sp_stats
from typing import Dict, Any


def two_sample_ttest(
    df: pd.DataFrame,
    value_col: str = "NO2_GT",
    group_col: str = "Temperature",
    split_method: str = "median",
) -> Dict[str, Any]:
    """Two-sample t-test: compare pollution on high vs low temperature days.

    H_0: mu_high_temp = mu_low_temp
    H_1: mu_high_temp != mu_low_temp

    Assumptions:
        - Independent samples (violated by time-series autocorrelation -- noted)
        - Approximate normality (CLT applies for large n)
        - Equal variance (Welch's t-test used to relax this)
    """
    group_series = pd.to_numeric(df[group_col], errors="coerce")
    threshold = group_series.median()
    group_high = df.loc[group_series >= threshold, value_col].dropna()
    group_low = df.loc[group_series < threshold, value_col].dropna()

    t_stat, p_value = sp_stats.ttest_ind(group_high, group_low, equal_var=False)

    return {
        "test": "Welch's Two-Sample t-test",
        "null_hypothesis": f"No difference in mean {value_col} between high/low {group_col} days",
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "group_high_mean": float(group_high.mean()),
        "group_low_mean": float(group_low.mean()),
        "group_high_n": int(len(group_high)),
        "group_low_n": int(len(group_low)),
        "significant_at_0.05": bool(p_value < 0.05),
        "caveat": (
            "Independence assumption is violated due to temporal autocorrelation "
            "in hourly time-series data. p-value may be anti-conservative."
        ),
    }


def seasonal_anova(
    df: pd.DataFrame,
    value_col: str = "NO2_GT",
    season_col: str = "Season",
) -> Dict[str, Any]:
    """One-way ANOVA: test pollution differences across seasons.

    H_0: mu_winter = mu_spring = mu_summer = mu_autumn
    H_1: At least one seasonal mean differs
    """
    groups = [g[value_col].dropna().values for _, g in df.groupby(season_col)]
    f_stat, p_value = sp_stats.f_oneway(*groups)

    season_means = df.groupby(season_col)[value_col].mean().to_dict()

    return {
        "test": "One-Way ANOVA",
        "null_hypothesis": f"Equal mean {value_col} across all seasons",
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "season_means": {str(k): float(v) for k, v in season_means.items()},
        "significant_at_0.05": bool(p_value < 0.05),
        "caveat": (
            "ANOVA assumes independent observations and homoscedasticity. "
            "Time-series autocorrelation violates independence; results should "
            "be interpreted cautiously."
        ),
    }


def chi_square_test(
    df: pd.DataFrame,
    pollution_col: str = "High_Pollution",
    humidity_col: str = "Rel_Humidity",
    bins: list | None = None,
    labels: list | None = None,
) -> Dict[str, Any]:
    """Chi-square test of independence: pollution threshold * humidity bucket.

    H_0: High_Pollution and Humidity_Bucket are independent
    H_1: There is an association
    """
    if bins is None:
        bins = [0, 30, 60, 100]
    if labels is None:
        labels = ["Low", "Medium", "High"]

    df = df.copy()
    df["Humidity_Bucket"] = pd.cut(df[humidity_col], bins=bins, labels=labels, include_lowest=True)

    contingency = pd.crosstab(df["Humidity_Bucket"], df[pollution_col])
    chi2, p_value, dof, expected = sp_stats.chi2_contingency(contingency)

    return {
        "test": "Chi-Square Test of Independence",
        "null_hypothesis": "High_Pollution and Humidity_Bucket are independent",
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "contingency_table": contingency.to_dict(),
        "significant_at_0.05": bool(p_value < 0.05),
        "caveat": (
            "Chi-square assumes independent observations. Consecutive hourly "
            "readings are autocorrelated, violating this assumption. The test "
            "may overstate significance."
        ),
    }
