"""Risk and threshold analysis: optimize decision boundary for cost-benefit."""

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from typing import Tuple


def threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    tp_benefit: float = 10,
    fp_cost: float = 3,
    fn_cost: float = 15,
    tn_benefit: float = 0,
    n_thresholds: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Sweep decision thresholds and compute F1 + expected value at each.

    Value = TP * Benefit - FP * Cost - FN * Cost_miss + TN * Benefit_baseline

    Returns:
        thresholds, f1_scores, expected_values, best_f1_threshold, best_ev_threshold
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    f1s = []
    evs = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        ev = tp * tp_benefit - fp * fp_cost - fn * fn_cost + tn * tn_benefit
        f1s.append(f1)
        evs.append(ev)

    f1s = np.array(f1s)
    evs = np.array(evs)

    best_f1_idx = np.argmax(f1s)
    best_ev_idx = np.argmax(evs)

    return (
        thresholds,
        f1s,
        evs,
        float(thresholds[best_f1_idx]),
        float(thresholds[best_ev_idx]),
    )
