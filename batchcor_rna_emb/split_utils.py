"""Утилиты для генерации и обработки split-колонок в AnnData.

Номенклатура:
    Колонка: ``Split_<TARGET_NAME>``
    Значения: ``"train"`` | ``"test"`` | ``np.nan`` (неаннотированные).
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from loguru import logger


def add_dummy_split(
    adata: ad.AnnData,
    col_name: str = "Split_dummy",
    train_frac: float = 0.60,
    test_frac: float = 0.20,
    seed: int = 42,
) -> ad.AnnData:
    """Add a dummy train/test/NaN split column to ``adata.obs``.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object (modified in-place).
    col_name : str
        Name of the new split column.
    train_frac : float
        Fraction of samples assigned to ``"train"``.
    test_frac : float
        Fraction of samples assigned to ``"test"``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ad.AnnData
        Same AnnData with ``adata.obs[col_name]`` added.

    Raises
    ------
    ValueError
        If ``train_frac + test_frac > 1.0``.
    """
    if train_frac + test_frac > 1.0:
        raise ValueError(
            f"train_frac + test_frac = {train_frac + test_frac} > 1.0"
        )

    nan_frac = 1.0 - train_frac - test_frac
    rng = np.random.RandomState(seed)
    n = adata.n_obs

    choices = rng.choice(
        ["train", "test", "_nan"],
        size=n,
        p=[train_frac, test_frac, nan_frac],
    )

    # Convert to pandas Categorical with NaN for unannotated
    split = pd.array(
        [c if c != "_nan" else pd.NA for c in choices],
        dtype=pd.CategoricalDtype(categories=["train", "test"]),
    )
    adata.obs[col_name] = split

    counts = pd.Series(choices).value_counts()
    logger.info(
        "Added split column '{}': train={}, test={}, NaN={}",
        col_name,
        counts.get("train", 0),
        counts.get("test", 0),
        counts.get("_nan", 0),
    )
    return adata


def get_split_masks(
    adata: ad.AnnData,
    split_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for train and test splits, ignoring NaN rows.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with ``adata.obs[split_col]``.
    split_col : str
        Name of the split column.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(train_mask, test_mask)`` — boolean arrays of length ``n_obs``.
        Rows with NaN in the split column are ``False`` in both masks.

    Raises
    ------
    KeyError
        If ``split_col`` not in ``adata.obs``.
    """
    if split_col not in adata.obs.columns:
        raise KeyError(f"Split column '{split_col}' not found in adata.obs")

    s = adata.obs[split_col].astype(str)
    train_mask = (s == "train").values
    test_mask = (s == "test").values

    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())
    n_nan = adata.n_obs - n_train - n_test
    logger.debug(
        "Split '{}': train={}, test={}, ignored(NaN)={}",
        split_col, n_train, n_test, n_nan,
    )
    return train_mask, test_mask
