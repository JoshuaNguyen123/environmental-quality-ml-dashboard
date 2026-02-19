"""Supervised learning: regression and classification model training."""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ===========================================================================
# REGRESSION
# ===========================================================================

def train_linear_regression(
    X_train: np.ndarray, y_train: np.ndarray
) -> Pipeline:
    """Ordinary Least Squares: y_hat = Xbeta + epsilon

    Assumptions:
        - Linearity between features and target
        - Homoscedasticity (constant variance of errors)
        - Independence of errors
        - No perfect multicollinearity
        - Normality of residuals (for valid inference)
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_random_forest_reg(
    X_train: np.ndarray, y_train: np.ndarray, params: dict | None = None
) -> Pipeline:
    """Random Forest Regression: y_hat = (1/T) Sum f_t(x)

    Non-parametric ensemble of decision trees with bagging.
    Strengths: nonlinear, captures interactions, robust to outliers.
    Weaknesses: less interpretable, may overfit with deep trees.
    """
    if params is None:
        params = {}
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(**params)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ===========================================================================
# CLASSIFICATION
# ===========================================================================

def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray, params: dict | None = None
) -> Pipeline:
    """Logistic Regression: P(y=1|x) = sigma(Xbeta)

    Assumptions:
        - Linear relationship between features and log-odds
        - No perfect multicollinearity
        - Independent observations (violated by time-series)
    """
    if params is None:
        params = {}
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(**params)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_random_forest_clf(
    X_train: np.ndarray, y_train: np.ndarray, params: dict | None = None
) -> Pipeline:
    """Random Forest Classifier: majority vote from T trees."""
    if params is None:
        params = {}
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(**params)),
    ])
    pipe.fit(X_train, y_train)
    return pipe
