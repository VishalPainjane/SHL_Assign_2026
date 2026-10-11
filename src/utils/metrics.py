"""
Evaluation Metrics
==================
Implements competition evaluation metrics:
- Root Mean Squared Error (RMSE) - Primary Metric
- Pearson Correlation (r) - Secondary Metric
- Spearman Rank Correlation (rho)
- Quadratic Weighted Kappa (QWK)
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Pearson Correlation coefficient."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2 or np.std(y_pred) < 1e-6 or np.std(y_true) < 1e-6:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Spearman Rank Correlation coefficient."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2 or np.std(y_pred) < 1e-6 or np.std(y_true) < 1e-6:
        return 0.0
    return float(spearmanr(y_true, y_pred)[0])


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Quadratic Weighted Kappa on half-integer rounded bins."""
    y_true_bins = np.clip(np.round(np.asarray(y_true) * 2.0).astype(int), 0, 10)
    y_pred_bins = np.clip(np.round(np.asarray(y_pred) * 2.0).astype(int), 0, 10)
    return float(cohen_kappa_score(y_true_bins, y_pred_bins, weights="quadratic"))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes full evaluation report dictionary."""
    return {
        "rmse": rmse(y_true, y_pred),
        "pearson": pearson_corr(y_true, y_pred),
        "spearman": spearman_corr(y_true, y_pred),
        "qwk": qwk(y_true, y_pred),
        "mae": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))),
    }
