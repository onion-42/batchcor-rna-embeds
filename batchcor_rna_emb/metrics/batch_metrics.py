"""Batch mixing metrics: kBET, iLISI, ASW batch, Graph Connectivity.

Uses scib-metrics for optimized implementations where available,
with sklearn fallback for Silhouette-based ASW.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc
from loguru import logger
from sklearn.metrics import silhouette_samples


def compute_kbet(
    adata: ad.AnnData,
    emb_key: str,
    batch_col: str,
    n_neighbors: int = 50,
) -> float:
    """
    Compute kBET acceptance rate (batch effect test).

    Higher = better batch mixing. Uses scib-metrics implementation.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with embedding and batch annotation.
    emb_key : str
        Key in ``.obsm`` for the embedding.
    batch_col : str
        Column in ``.obs`` for batch labels.
    n_neighbors : int
        Number of neighbors for kBET test.

    Returns
    -------
    float
        kBET acceptance rate in [0, 1].
    """
    from scib_metrics import kbet as _kbet

    # scib-metrics requires a kNN graph; compute on embedding
    adata_tmp = adata.copy()
    sc.pp.neighbors(adata_tmp, use_rep=emb_key, n_neighbors=n_neighbors)

    score = _kbet(
        adata_tmp.obsp["connectivities"],
        adata_tmp.obs[batch_col].to_numpy(),
    )
    result = float(np.nanmean(score))
    logger.debug("kBET acceptance rate: {:.4f}", result)
    return result


def compute_graph_connectivity(
    adata: ad.AnnData,
    batch_col: str,
    n_neighbors: int = 30,
    emb_key: str | None = None,
) -> float:
    """
    Compute graph connectivity score.

    For each batch label, measure fraction of samples in the largest
    connected component of the kNN subgraph. Average across batches.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with kNN graph (or embedding to build one).
    batch_col : str
        Column in ``.obs`` for batch labels.
    n_neighbors : int
        Number of neighbors.
    emb_key : str or None
        Embedding key. If None, uses pre-computed kNN graph.

    Returns
    -------
    float
        Graph connectivity score in [0, 1]. Higher = better.
    """
    from scib_metrics import graph_connectivity as _gc

    adata_tmp = adata.copy()
    if emb_key is not None:
        sc.pp.neighbors(adata_tmp, use_rep=emb_key, n_neighbors=n_neighbors)

    score = _gc(
        adata_tmp.obsp["connectivities"],
        adata_tmp.obs[batch_col].to_numpy(),
    )
    result = float(score)
    logger.debug("Graph connectivity: {:.4f}", result)
    return result


def compute_ilisi(
    adata: ad.AnnData,
    emb_key: str,
    batch_col: str,
    n_neighbors: int = 30,
) -> float:
    """
    Compute integration LISI (iLISI) for batch mixing.

    Higher iLISI = better batch mixing (ideal = number of batches).
    Rescaled to [0, 1] by ``(median - 1) / (n_batches - 1)``.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with embedding and batch annotation.
    emb_key : str
        Key in ``.obsm`` for the embedding.
    batch_col : str
        Column in ``.obs`` for batch labels.
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    float
        Scaled iLISI in [0, 1].
    """
    from scib_metrics import ilisi_knn

    adata_tmp = adata.copy()
    sc.pp.neighbors(adata_tmp, use_rep=emb_key, n_neighbors=n_neighbors)

    scores = ilisi_knn(
        adata_tmp.obsp["connectivities"],
        adata_tmp.obs[batch_col].to_numpy(),
    )
    n_batches = adata.obs[batch_col].nunique()
    # rescale: (median - 1) / (n_batches - 1), clamped to [0,1]
    raw_median = float(np.nanmedian(scores))
    scaled = (raw_median - 1.0) / max(n_batches - 1, 1)
    result = float(np.clip(scaled, 0.0, 1.0))
    logger.debug("iLISI scaled: {:.4f} (raw median={:.4f})", result, raw_median)
    return result


def compute_asw_batch(
    adata: ad.AnnData,
    emb_key: str,
    batch_col: str,
) -> float:
    """
    Compute Average Silhouette Width for batch (ASW batch).

    ``ASW_batch = 1 - |mean(silhouette_samples)|``, scaled to [0, 1].
    Higher = better batch mixing (batch clusters not separated).

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with embedding and batch annotation.
    emb_key : str
        Key in ``.obsm`` for the embedding.
    batch_col : str
        Column in ``.obs`` for batch labels.

    Returns
    -------
    float
        ASW batch score in [0, 1].
    """
    X = np.asarray(adata.obsm[emb_key])
    labels = adata.obs[batch_col].to_numpy()

    n_unique = len(np.unique(labels))
    if n_unique < 2:
        logger.warning("ASW batch: fewer than 2 batches, returning 1.0")
        return 1.0

    sil = silhouette_samples(X, labels, metric="euclidean")
    asw = 1.0 - abs(float(np.mean(sil)))
    result = float(np.clip(asw, 0.0, 1.0))
    logger.debug("ASW batch: {:.4f}", result)
    return result
