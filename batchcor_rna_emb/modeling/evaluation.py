"""Evaluation metrics: Youden threshold, binary classifier evaluation, C-index.

Ported from eury_main utils_metrics with adaptations for this project.
"""

from __future__ import annotations

from lifelines import CoxPHFitter
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def youden_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    default_thr: float = 0.5,
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    """
    Compute robust Youden index cut-off with degenerate case handling.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted positive probabilities.
    default_thr : float
        Fallback threshold when a meaningful cut-off cannot be derived.
    eps : float
        Safety margin for boundary thresholds.

    Returns
    -------
    tuple[float, float, float]
        (threshold, true_positive_rate, false_positive_rate).
        TPR/FPR are ``nan`` when not calculable.
    """
    unique = np.unique(y_prob)

    # degenerate: constant predictor
    if unique.size == 1:
        return default_thr, np.nan, np.nan

    # degenerate: binary 0/1 scores (perfect separator)
    if unique.size == 2 and set(unique) <= {0.0, 1.0}:
        tpr_val = float((y_true == 1).mean())
        return 0.5, tpr_val, 0.0

    # general case
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    good = np.isfinite(thr) & (thr > eps) & (thr < 1 - eps)
    if not np.any(good):
        return default_thr, np.nan, np.nan

    fpr, tpr, thr = fpr[good], tpr[good], thr[good]
    j_idx = int(np.argmax(tpr - fpr))
    return float(thr[j_idx]), float(tpr[j_idx]), float(fpr[j_idx])


def evaluate_binary_classifier(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> pd.Series:
    """
    Evaluate a binary classifier with comprehensive metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    probabilities : np.ndarray
        Predicted probabilities for the positive class.
    threshold : float
        Decision threshold for binary predictions.

    Returns
    -------
    pd.Series
        Metrics: f1, f1_weighted, f1_macro, pr_auc, roc_auc,
        accuracy, balanced_accuracy, precision, recall.
    """
    y_pred = (probabilities >= threshold).astype(int)
    n_classes = len(np.unique(y_true))

    metrics = {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "pr_auc": average_precision_score(y_true, probabilities)
        if n_classes > 1
        else np.nan,
        "roc_auc": roc_auc_score(y_true, probabilities) if n_classes > 1 else np.nan,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    return pd.Series(metrics)


def compute_c_index(
    design_df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariate_cols: list[str] | None = None,
    penalizer: float = 0.01,
) -> float:
    """
    Compute concordance index using Cox Proportional Hazards model.

    Parameters
    ----------
    design_df : pd.DataFrame
        DataFrame with survival data and covariates.
    duration_col : str
        Column for time-to-event (e.g. OS, PFS).
    event_col : str
        Column for event indicator (1=event, 0=censored).
    covariate_cols : list[str] or None
        Columns to use as covariates. If None, uses all columns
        except ``duration_col`` and ``event_col``.
    penalizer : float
        L2 penalizer for CoxPH.

    Returns
    -------
    float
        Concordance index (C-index). Ideal >= 0.75 for oncology.

    Raises
    ------
    ValueError
        If required columns are missing or data is insufficient.
    """
    for col in [duration_col, event_col]:
        if col not in design_df.columns:
            raise ValueError(f"Column '{col}' not found in design_df")

    if covariate_cols is not None:
        cols_to_use = covariate_cols + [duration_col, event_col]
    else:
        cols_to_use = list(design_df.columns)

    df = design_df[cols_to_use].dropna()

    if df.shape[0] < 10:
        logger.warning("Too few samples ({}) for Cox PH, returning NaN", df.shape[0])
        return np.nan

    if df[event_col].nunique() < 2:
        logger.warning("No events or all events in '{}', returning NaN", event_col)
        return np.nan

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(df, duration_col=duration_col, event_col=event_col)
    except (ValueError, np.linalg.LinAlgError) as e:
        logger.error("CoxPH fitting failed: {}", e)
        return np.nan

    c_index = float(cph.concordance_index_)
    logger.debug("C-index: {:.4f} ({} samples)", c_index, df.shape[0])
    return c_index
