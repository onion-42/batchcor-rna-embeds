"""Biological signal preservation metrics: cLISI, Silhouette bio, NMI, ARI.

'Bio labels' in this project = diagnosis (cancer type), NOT cell types.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc
from loguru import logger
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
)


def compute_clisi(
    adata: ad.AnnData,
    emb_key: str,
    label_col: str,
    n_neighbors: int = 30,
) -> float:
    """
    Compute cell-type (diagnosis) LISI (cLISI).

    Lower cLISI = better biological signal preservation.
    Rescaled: ``1 - (median - 1) / (n_labels - 1)``, clamped to [0, 1].

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with embedding and bio labels.
    emb_key : str
        Key in ``.obsm`` for the embedding.
    label_col : str
        Column in ``.obs`` for biological labels (e.g. diagnosis).
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    float
        Scaled cLISI in [0, 1]. Higher = better preservation.
    """
    from scib_metrics import clisi_knn

    adata_tmp = adata.copy()
    sc.pp.neighbors(adata_tmp, use_rep=emb_key, n_neighbors=n_neighbors)

    scores = clisi_knn(
        adata_tmp.obsp["connectivities"],
        adata_tmp.obs[label_col].to_numpy(),
    )
    n_labels = adata.obs[label_col].nunique()
    raw_median = float(np.nanmedian(scores))
    # cLISI: ideal = 1 (pure clusters), rescale and invert
    scaled = 1.0 - (raw_median - 1.0) / max(n_labels - 1, 1)
    result = float(np.clip(scaled, 0.0, 1.0))
    logger.debug("cLISI scaled: {:.4f} (raw median={:.4f})", result, raw_median)
    return result


def compute_silhouette_bio(
    adata: ad.AnnData,
    emb_key: str,
    label_col: str,
) -> float:
    """
    Compute Silhouette score for biological labels.

    Higher = better biological cluster separation.
    Rescaled from [-1, 1] to [0, 1]: ``(score + 1) / 2``.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with embedding and bio labels.
    emb_key : str
        Key in ``.obsm`` for the embedding.
    label_col : str
        Column in ``.obs`` for biological labels.

    Returns
    -------
    float
        Silhouette bio score in [0, 1].
    """
    X = np.asarray(adata.obsm[emb_key])
    labels = adata.obs[label_col].to_numpy()

    n_unique = len(np.unique(labels))
    if n_unique < 2:
        logger.warning("Silhouette bio: fewer than 2 labels, returning 0.0")
        return 0.0

    sil = silhouette_samples(X, labels, metric="euclidean")
    # rescale from [-1, 1] to [0, 1]
    result = float(np.clip((np.mean(sil) + 1.0) / 2.0, 0.0, 1.0))
    logger.debug("Silhouette bio: {:.4f}", result)
    return result


def compute_nmi(
    adata: ad.AnnData,
    label_col: str,
    cluster_key: str = "leiden",
) -> float:
    """
    Compute Normalized Mutual Information between clusters and bio labels.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with cluster assignment and bio labels in ``.obs``.
    label_col : str
        Column for biological labels.
    cluster_key : str
        Column for cluster assignments.

    Returns
    -------
    float
        NMI in [0, 1].

    Raises
    ------
    KeyError
        If ``label_col`` or ``cluster_key`` not in ``.obs``.
    """
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"Cluster column '{cluster_key}' not found in .obs")

    labels_true = adata.obs[label_col].to_numpy()
    labels_pred = adata.obs[cluster_key].to_numpy()

    result = float(normalized_mutual_info_score(labels_true, labels_pred))
    logger.debug("NMI: {:.4f}", result)
    return result


def compute_ari(
    adata: ad.AnnData,
    label_col: str,
    cluster_key: str = "leiden",
) -> float:
    """
    Compute Adjusted Rand Index between clusters and bio labels.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with cluster assignment and bio labels in ``.obs``.
    label_col : str
        Column for biological labels.
    cluster_key : str
        Column for cluster assignments.

    Returns
    -------
    float
        ARI in [-1, 1] (typically [0, 1] for good clusterings).

    Raises
    ------
    KeyError
        If ``label_col`` or ``cluster_key`` not in ``.obs``.
    """
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"Cluster column '{cluster_key}' not found in .obs")

    labels_true = adata.obs[label_col].to_numpy()
    labels_pred = adata.obs[cluster_key].to_numpy()

    result = float(adjusted_rand_score(labels_true, labels_pred))
    logger.debug("ARI: {:.4f}", result)
    return result


def run_leiden_clustering(
    adata: ad.AnnData,
    emb_key: str,
    resolution: float = 1.0,
    n_neighbors: int = 30,
    cluster_key: str = "leiden",
) -> ad.AnnData:
    """
    Run Leiden clustering on an embedding.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with embedding.
    emb_key : str
        Key in ``.obsm`` for the embedding.
    resolution : float
        Leiden resolution parameter.
    n_neighbors : int
        Number of neighbors for graph construction.
    cluster_key : str
        Key to store cluster assignments in ``.obs``.

    Returns
    -------
    ad.AnnData
        Modified AnnData with ``.obs[cluster_key]``.
    """
    sc.pp.neighbors(adata, use_rep=emb_key, n_neighbors=n_neighbors)
    sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key)
    n_clusters = adata.obs[cluster_key].nunique()
    logger.info("Leiden clustering (res={:.2f}): {} clusters", resolution, n_clusters)
    return adata
