"""Metric aggregation: geometric mean and comparison table builder."""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def geometric_mean(values: list[float]) -> float:
    """
    Compute the geometric mean of a list of non-negative values.

    Zero values result in zero geometric mean (as intended for penalization).

    Parameters
    ----------
    values : list[float]
        Non-negative metric values.

    Returns
    -------
    float
        Geometric mean.

    Raises
    ------
    ValueError
        If any value is negative.
    """
    arr = np.array(values, dtype=np.float64)
    if np.any(arr < 0):
        raise ValueError(f"Geometric mean requires non-negative values, got: {values}")
    if len(arr) == 0:
        return 0.0
    if np.any(arr == 0):
        return 0.0
    result = float(np.exp(np.mean(np.log(arr))))
    return result


def build_comparison_table(
    results: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """
    Build a comparison table from nested metric results.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        Outer key = method name (e.g. 'Raw', 'Harmony', 'DANN').
        Inner dict = metric_name -> value.

    Returns
    -------
    pd.DataFrame
        Table with methods as rows and metrics as columns,
        plus AvgBATCH and AvgBio columns if constituent metrics exist.
    """
    df = pd.DataFrame(results).T
    df.index.name = "method"

    # compute aggregate scores if constituent metrics present
    batch_cols = ["kBET", "graph_connectivity", "iLISI", "ASW_batch"]
    bio_cols = ["cLISI", "silhouette_bio", "NMI", "ARI"]

    available_batch = [c for c in batch_cols if c in df.columns]
    available_bio = [c for c in bio_cols if c in df.columns]

    if available_batch:
        df["AvgBATCH"] = df[available_batch].apply(
            lambda row: geometric_mean(row.tolist()), axis=1,
        )
        logger.info("AvgBATCH computed from: {}", available_batch)

    if available_bio:
        df["AvgBio"] = df[available_bio].apply(
            lambda row: geometric_mean(row.tolist()), axis=1,
        )
        logger.info("AvgBio computed from: {}", available_bio)

    return df
