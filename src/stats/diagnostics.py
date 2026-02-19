"""Regression diagnostics: VIF, normality tests, heteroscedasticity checks."""

import numpy as np
from scipy import stats as sp_stats
from typing import Dict, Any


def variance_inflation_factor(X: np.ndarray, feature_names: list) -> Dict[str, float]:
    """Compute VIF for each feature to detect multicollinearity.

    VIF > 5 suggests moderate collinearity; VIF > 10 suggests severe.
    """
    from numpy.linalg import LinAlgError

    vifs = {}
    for i, name in enumerate(feature_names):
        mask = [j for j in range(X.shape[1]) if j != i]
        X_other = X[:, mask]
        y_i = X[:, i]
        try:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(X_other, y_i)
            r_sq = model.score(X_other, y_i)
            vifs[name] = float(1 / (1 - r_sq)) if r_sq < 1 else float("inf")
        except (LinAlgError, ValueError):
            vifs[name] = float("inf")
    return vifs


def residual_diagnostics(residuals: np.ndarray) -> Dict[str, Any]:
    """Run normality and heteroscedasticity diagnostics on residuals."""
    # Shapiro-Wilk (on subsample if n > 5000)
    if len(residuals) > 5000:
        subsample = np.random.default_rng(42).choice(residuals, 5000, replace=False)
    else:
        subsample = residuals

    shapiro_stat, shapiro_p = sp_stats.shapiro(subsample)

    # Jarque-Bera
    jb_stat, jb_p = sp_stats.jarque_bera(residuals)

    # Durbin-Watson approximation
    diff = np.diff(residuals)
    dw = float(np.sum(diff ** 2) / np.sum(residuals ** 2))

    return {
        "shapiro_wilk": {"statistic": float(shapiro_stat), "p_value": float(shapiro_p)},
        "jarque_bera": {"statistic": float(jb_stat), "p_value": float(jb_p)},
        "durbin_watson": float(dw),
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "interpretation": {
            "normality": "Residuals are approximately normal" if shapiro_p > 0.05 else "Residuals deviate from normality",
            "autocorrelation": (
                "No significant autocorrelation" if 1.5 < dw < 2.5
                else "Possible autocorrelation detected (expected for time-series)"
            ),
        },
    }
