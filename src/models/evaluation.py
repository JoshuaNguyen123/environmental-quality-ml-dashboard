"""Model evaluation: unified metrics for regression and classification."""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from typing import Dict, Any, Tuple


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics: RMSE, MAE, R^2."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Compute classification metrics: AUC, precision, recall, F1, confusion matrix."""
    result = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        result["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        result["pr_auc"] = float(average_precision_score(y_true, y_proba))
        result["log_loss"] = float(log_loss(y_true, y_proba))
    return result


def get_roc_data(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (fpr, tpr, auc) for ROC curve plotting."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)
    return fpr, tpr, auc_val


def get_pr_data(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (precision, recall, ap) for PR curve plotting."""
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    return prec, rec, ap


def get_calibration_data(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fraction_of_positives, mean_predicted_value)."""
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    return frac_pos, mean_pred


def compute_permutation_importance(
    model, X: np.ndarray, y: np.ndarray, feature_names: list, n_repeats: int = 10
) -> Dict[str, float]:
    """Permutation importance: model-agnostic feature importance."""
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    return {name: float(imp) for name, imp in zip(feature_names, result.importances_mean)}
