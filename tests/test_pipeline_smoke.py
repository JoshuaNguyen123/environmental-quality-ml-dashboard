"""End-to-end smoke test for the data → features → split → train pipeline.

Marked ``slow`` — skipped by default, run on PRs to main via CI:

    pytest -m slow
"""

from __future__ import annotations

import pytest

from src.data.clean import clean
from src.data.ingest import generate_synthetic_air_quality
from src.data.split import chronological_split
from src.features.engineer import engineer_features, get_feature_columns
from src.features.preprocessing import prepare_xy
from src.models.evaluation import classification_metrics, regression_metrics
from src.models.supervised import (
    train_linear_regression,
    train_logistic_regression,
)

pytestmark = pytest.mark.slow


def test_pipeline_smoke_runs_end_to_end():
    """Synthetic data → clean → features → split → train → predict on a tiny subset."""
    df_raw = generate_synthetic_air_quality(n_hours=600, seed=123)
    df_clean = clean(df_raw)
    assert len(df_clean) > 100, "cleaning should retain most synthetic rows"

    df_feat = engineer_features(df_clean, regulatory_threshold=120)
    assert "High_Pollution" in df_feat.columns

    train, val, test = chronological_split(df_feat)
    assert len(train) > 0 and len(val) > 0 and len(test) > 0

    feature_cols = get_feature_columns(df_feat)
    X_train, y_reg_train = prepare_xy(train, feature_cols, "NO2_GT")
    X_test, y_reg_test = prepare_xy(test, feature_cols, "NO2_GT")
    _, y_clf_train = prepare_xy(train, feature_cols, "High_Pollution")
    _, y_clf_test = prepare_xy(test, feature_cols, "High_Pollution")

    # Regression: linear model should produce finite predictions.
    reg = train_linear_regression(X_train, y_reg_train)
    reg_pred = reg.predict(X_test)
    reg_metrics = regression_metrics(y_reg_test, reg_pred)
    assert reg_metrics["rmse"] >= 0
    assert -2.0 <= reg_metrics["r2"] <= 1.0  # may be weak on tiny sample

    # Classification: logistic model should produce valid probabilities.
    clf = train_logistic_regression(X_train, y_clf_train)
    clf_pred = clf.predict(X_test)
    clf_proba = clf.predict_proba(X_test)[:, 1]
    clf_metrics = classification_metrics(y_clf_test, clf_pred, clf_proba)
    assert 0.0 <= clf_metrics["f1"] <= 1.0
    assert 0.0 <= clf_metrics["roc_auc"] <= 1.0
