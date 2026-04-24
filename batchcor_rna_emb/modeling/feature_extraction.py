"""PCA knee detection and pipeline fitting for expression data.

Ported from eury_main utils_modeling.detect_PCA_knee with adaptations.
"""
from __future__ import annotations

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def detect_pca_knee(
    data: Union[pd.DataFrame, np.ndarray],
    n_components: int = 100,
    seed: int = 42,
    plot: bool = True,
) -> int:
    """
    Detect the knee/elbow in cumulative PCA explained variance.

    Scales data with StandardScaler, fits PCA, and uses KneeLocator
    to find the optimal number of components.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Input data with samples as rows and features as columns.
        Gene expressions should be in log2(TPM + 1) scale.
    n_components : int
        Maximum number of PCA components to compute.
    seed : int
        Random seed for PCA reproducibility.
    plot : bool
        If True, plot the cumulative explained variance with knee marker.

    Returns
    -------
    int
        The 1-based number of components at the knee.
        Returns ``n_components`` if no knee is detected.
    """
    n_samp, n_feat = data.shape if isinstance(data, np.ndarray) else data.values.shape
    n_components_ = min(n_components, n_feat, n_samp - 1)

    if n_components_ < n_components:
        logger.info(
            "n_components clipped to {} = min(requested, n_feat, n_samp-1)",
            n_components_,
        )

    scaler = StandardScaler()
    pca = PCA(n_components=n_components_, random_state=seed)

    scaled_data = scaler.fit_transform(data)
    pca.fit(scaled_data)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100

    kl = KneeLocator(
        x=np.arange(0, n_components_),
        y=cumulative_variance,
        curve="concave",
        direction="increasing",
    )
    knee_idx = kl.knee

    if knee_idx is None:
        logger.info("No knee detected by KneeLocator. Returning n_components={}", n_components_)
        return n_components_

    knee = knee_idx + 1  # 1-based
    knee_variance = round(float(cumulative_variance[knee_idx]), 2)
    logger.info("Elbow at component: {}, explained variance: {}%", knee, knee_variance)

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(
            np.arange(1, len(cumulative_variance) + 1),
            cumulative_variance,
            marker="o", linestyle="-", color="g", markersize=3,
        )
        plt.axvline(
            knee, color="black", linestyle="--",
            label=f"n_comp = {knee}, explained var = {knee_variance}%",
        )
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance (%)")
        plt.title("PCA Knee Detection")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return knee


def fit_pca_pipeline(
    X_train: np.ndarray,
    n_components: int,
    seed: int = 42,
) -> tuple[StandardScaler, PCA]:
    """
    Fit StandardScaler + PCA on training data.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    n_components : int
        Number of PCA components (typically from ``detect_pca_knee``).
    seed : int
        Random seed.

    Returns
    -------
    tuple[StandardScaler, PCA]
        Fitted scaler and PCA transformer.
    """
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=seed)

    X_scaled = scaler.fit_transform(X_train)
    pca.fit(X_scaled)

    total_var = float(np.sum(pca.explained_variance_ratio_) * 100)
    logger.info(
        "PCA pipeline fitted: {} components, {:.1f}% variance explained",
        n_components, total_var,
    )
    return scaler, pca


def transform_pca_pipeline(
    X: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    """
    Transform data through a fitted StandardScaler + PCA pipeline.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix.
    scaler : StandardScaler
        Fitted scaler.
    pca : PCA
        Fitted PCA transformer.

    Returns
    -------
    np.ndarray
        Transformed data of shape ``(n_samples, n_components)``.
    """
    return pca.transform(scaler.transform(X))
