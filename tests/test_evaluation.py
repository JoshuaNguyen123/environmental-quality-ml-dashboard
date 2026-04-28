"""Tests for evaluation metrics on hand-crafted inputs."""

from __future__ import annotations

import math

import numpy as np

from src.models.evaluation import (
    classification_metrics,
    get_pr_data,
    get_roc_data,
    regression_metrics,
)


def test_regression_metrics_perfect_fit():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y.copy())
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0


def test_regression_metrics_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.5, 2.5, 3.5])
    m = regression_metrics(y_true, y_pred)
    assert math.isclose(m["mae"], 0.5, rel_tol=1e-9)
    assert math.isclose(m["rmse"], 0.5, rel_tol=1e-9)


def test_classification_metrics_perfect():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])
    m = classification_metrics(y_true, y_pred, y_proba)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0


def test_classification_metrics_all_negative_predictions():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    m = classification_metrics(y_true, y_pred)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_roc_data_returns_finite_arrays():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.6, 0.3, 0.8])
    fpr, tpr, auc = get_roc_data(y_true, y_proba)
    assert len(fpr) == len(tpr) >= 2
    assert 0.0 <= auc <= 1.0
    assert auc == 1.0  # perfect separation


def test_pr_data_basic_shape():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.3, 0.7])
    prec, rec, ap = get_pr_data(y_true, y_proba)
    assert len(prec) == len(rec)
    assert 0.0 <= ap <= 1.0
