"""Stress test split logic: Sanity Check, Weak OOD, True OOD.

Split levels:
  - Sanity: train/test from one cohort (no batch mixing).
  - Weak OOD: train/test from all cohorts (same batches, different patients).
  - True OOD: train on subset of cohorts, test on completely held-out cohorts.
"""
from __future__ import annotations

from typing import NamedTuple

import anndata as ad
import numpy as np
import pandas as pd
from loguru import logger

from batchcor_rna_emb.config import BATCH_COL, SPLIT_PREFIX, TARGET_PREFIX


class StressTestSplit(NamedTuple):
    """Container for a single stress test split.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    X_test : np.ndarray
        Test feature matrix.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    level : str
        Stress level name ('sanity', 'weak_ood', 'true_ood').
    desc : str
        Human-readable description.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    level: str
    desc: str


def prepare_stress_test_splits(
    adata: ad.AnnData,
    target: str,
    holdout_cohorts: list[str],
    batch_col: str = BATCH_COL,
    sanity_cohort: str | None = None,
) -> dict[str, StressTestSplit]:
    """
    Prepare all three stress test split levels.

    Parameters
    ----------
    adata : ad.AnnData
        Combined AnnData with ``.X`` gene expression matrix,
        ``.obs[Split_{target}]`` (train/test), ``.obs[Target_{target}]`` (0/1),
        and ``.obs[batch_col]``.
    target : str
        Target name (e.g. 'Response_wo_SD'). Used to find
        ``Split_{target}`` and ``Target_{target}`` columns.
    holdout_cohorts : list[str]
        Cohort names (batch labels) to hold out entirely for True OOD.
    batch_col : str
        Column in ``.obs`` for batch/cohort labels.
    sanity_cohort : str or None
        Specific cohort for Sanity Check. If None, uses the largest
        non-holdout cohort.

    Returns
    -------
    dict[str, StressTestSplit]
        Keys: ``'sanity'``, ``'weak_ood'``, ``'true_ood'``.

    Raises
    ------
    KeyError
        If required columns are missing from ``.obs``.
    ValueError
        If no samples available for a split level.
    """
    split_col = f"{SPLIT_PREFIX}{target}"
    target_col = f"{TARGET_PREFIX}{target}"

    for col in [split_col, target_col, batch_col]:
        if col not in adata.obs.columns:
            raise KeyError(f"Required column '{col}' not found in adata.obs")

    X = _get_expression_matrix(adata)
    obs = adata.obs

    splits: dict[str, StressTestSplit] = {}

    # --- Sanity Check: single cohort, no batch mixing ---
    non_holdout = obs[~obs[batch_col].isin(holdout_cohorts)]
    if sanity_cohort is None:
        sanity_cohort = non_holdout[batch_col].value_counts().idxmax()

    sanity_mask = obs[batch_col] == sanity_cohort
    sanity_obs = obs[sanity_mask]
    sanity_X = X[sanity_mask.values]

    train_mask_s = sanity_obs[split_col] == "train"
    test_mask_s = sanity_obs[split_col] == "test"

    if train_mask_s.sum() > 0 and test_mask_s.sum() > 0:
        splits["sanity"] = StressTestSplit(
            X_train=sanity_X[train_mask_s.values],
            X_test=sanity_X[test_mask_s.values],
            y_train=sanity_obs.loc[train_mask_s, target_col].to_numpy().astype(int),
            y_test=sanity_obs.loc[test_mask_s, target_col].to_numpy().astype(int),
            level="sanity",
            desc=f"Sanity Check: cohort='{sanity_cohort}'",
        )
        logger.info(
            "Sanity split: train={}, test={}, cohort='{}'",
            train_mask_s.sum(), test_mask_s.sum(), sanity_cohort,
        )
    else:
        logger.warning("Sanity split: insufficient samples for cohort '{}'", sanity_cohort)

    # --- Weak OOD: all non-holdout cohorts, same Split column ---
    weak_mask = ~obs[batch_col].isin(holdout_cohorts)
    weak_obs = obs[weak_mask]
    weak_X = X[weak_mask.values]

    train_mask_w = weak_obs[split_col] == "train"
    test_mask_w = weak_obs[split_col] == "test"

    if train_mask_w.sum() > 0 and test_mask_w.sum() > 0:
        splits["weak_ood"] = StressTestSplit(
            X_train=weak_X[train_mask_w.values],
            X_test=weak_X[test_mask_w.values],
            y_train=weak_obs.loc[train_mask_w, target_col].to_numpy().astype(int),
            y_test=weak_obs.loc[test_mask_w, target_col].to_numpy().astype(int),
            level="weak_ood",
            desc="Weak OOD: all non-holdout cohorts, Split train/test",
        )
        logger.info(
            "Weak OOD split: train={}, test={}",
            train_mask_w.sum(), test_mask_w.sum(),
        )
    else:
        logger.warning("Weak OOD split: insufficient samples")

    # --- True OOD: train on non-holdout, test on holdout cohorts ---
    train_mask_t = ~obs[batch_col].isin(holdout_cohorts) & (obs[split_col] == "train")
    test_mask_t = obs[batch_col].isin(holdout_cohorts)

    if train_mask_t.sum() > 0 and test_mask_t.sum() > 0:
        # filter test samples with valid target
        test_valid = test_mask_t & obs[target_col].notna()
        if test_valid.sum() > 0:
            splits["true_ood"] = StressTestSplit(
                X_train=X[train_mask_t.values],
                X_test=X[test_valid.values],
                y_train=obs.loc[train_mask_t, target_col].to_numpy().astype(int),
                y_test=obs.loc[test_valid, target_col].to_numpy().astype(int),
                level="true_ood",
                desc=f"True OOD: holdout cohorts={holdout_cohorts}",
            )
            logger.info(
                "True OOD split: train={}, test={}, holdout={}",
                train_mask_t.sum(), test_valid.sum(), holdout_cohorts,
            )
        else:
            logger.warning("True OOD: no test samples with valid target")
    else:
        logger.warning("True OOD split: insufficient samples")

    return splits


def _get_expression_matrix(adata: ad.AnnData) -> np.ndarray:
    """
    Extract dense expression matrix from AnnData.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with ``.X``.

    Returns
    -------
    np.ndarray
        Dense expression matrix.
    """
    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        return np.asarray(X.toarray(), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def log_split_summary(splits: dict[str, StressTestSplit]) -> pd.DataFrame:
    """
    Log summary of stress test splits.

    Parameters
    ----------
    splits : dict[str, StressTestSplit]
        Prepared splits from ``prepare_stress_test_splits``.

    Returns
    -------
    pd.DataFrame
        Summary table with sample counts and class balance.
    """
    rows = []
    for name, s in splits.items():
        n_train_pos = int(s.y_train.sum())
        n_test_pos = int(s.y_test.sum())
        rows.append({
            "level": name,
            "n_train": len(s.y_train),
            "n_test": len(s.y_test),
            "train_pos_rate": n_train_pos / max(len(s.y_train), 1),
            "test_pos_rate": n_test_pos / max(len(s.y_test), 1),
            "description": s.desc,
        })

    df = pd.DataFrame(rows)
    logger.info("Stress test split summary:\n{}", df.to_string(index=False))
    return df
