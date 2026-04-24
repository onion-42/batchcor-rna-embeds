"""Harmony batch correction: stage-1 + BackTrack stage-2 integration.

Refactored from references/harmony_backtrack.ipynb into reusable functions
with loguru logging, type hints, and NumPy docstrings.
"""
from __future__ import annotations

import anndata as ad
import harmonypy as hm
import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

EPS: float = 1e-8


# ---------------------------------------------------------------------------
# Stage-1: vanilla Harmony on train-only
# ---------------------------------------------------------------------------

def run_harmony_stage1(
    adata: ad.AnnData,
    embedding_key: str,
    batch_col: str = "batch",
    max_iter: int = 10,
) -> np.ndarray:
    """
    Run Harmony stage-1 on a single embedding matrix, correcting for batch.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with ``.obsm[embedding_key]`` and ``.obs[batch_col]``.
    embedding_key : str
        Key in ``.obsm`` pointing to the raw embedding matrix.
    batch_col : str
        Column in ``.obs`` containing batch labels.
    max_iter : int
        Maximum Harmony iterations.

    Returns
    -------
    np.ndarray
        Corrected embedding matrix of shape ``(n_obs, n_features)``.

    Raises
    ------
    KeyError
        If ``embedding_key`` not in ``.obsm`` or ``batch_col`` not in ``.obs``.
    """
    if embedding_key not in adata.obsm:
        raise KeyError(f"obsm key '{embedding_key}' not found in AnnData")
    if batch_col not in adata.obs.columns:
        raise KeyError(f"obs column '{batch_col}' not found in AnnData")

    data_mat = np.asarray(adata.obsm[embedding_key], dtype=np.float32)
    metadata = adata.obs[[batch_col]].copy()

    logger.info(
        "Harmony stage-1: {} samples x {} dims, batch_col='{}', max_iter={}",
        data_mat.shape[0], data_mat.shape[1], batch_col, max_iter,
    )

    ho = hm.run_harmony(
        data_mat=data_mat,
        meta_data=metadata,
        vars_use=[batch_col],
        max_iter_harmony=max_iter,
    )

    corrected = np.asarray(ho.Z_corr, dtype=np.float32)
    # harmonypy returns (n_features, n_obs) — transpose if needed
    if corrected.shape[0] != data_mat.shape[0] and corrected.shape[1] == data_mat.shape[0]:
        corrected = corrected.T

    logger.info("Harmony stage-1 complete. Output shape: {}", corrected.shape)
    return corrected


# ---------------------------------------------------------------------------
# BackTrack helpers
# ---------------------------------------------------------------------------

def _make_combined(
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    batch_col: str,
    split_col: str,
    obsm_keys: tuple[str, ...],
) -> ad.AnnData:
    """
    Build a lightweight combined AnnData for stage-2 Harmony.

    Parameters
    ----------
    adata_train : ad.AnnData
        Training AnnData.
    adata_test : ad.AnnData
        Test AnnData.
    batch_col : str
        Batch column name in ``.obs``.
    split_col : str
        Split column name in ``.obs``.
    obsm_keys : tuple[str, ...]
        Keys to carry from ``.obsm``.

    Returns
    -------
    ad.AnnData
        Combined AnnData with ``obs['combined_batch']`` and ``uns['rows']``.
    """
    n_tr, n_te = adata_train.n_obs, adata_test.n_obs

    obs = pd.concat([adata_train.obs, adata_test.obs], axis=0)
    split_str = obs[split_col].astype(str) if split_col in obs.columns else pd.Series(
        ["unknown"] * len(obs), index=obs.index
    )
    batch_str = obs[batch_col].astype(str) if batch_col in obs.columns else pd.Series(
        ["NA"] * len(obs), index=obs.index
    )

    obs["combined_batch"] = (
        batch_str.fillna("NA") + "_" + split_str.fillna("NA")
    ).astype("category")

    # placeholder X to avoid copying gene matrix
    X = np.zeros((n_tr + n_te, 1), dtype=np.float32)

    obsm: dict[str, np.ndarray] = {}
    for k in obsm_keys:
        if k in adata_train.obsm and k in adata_test.obsm:
            obsm[k] = np.vstack([
                np.asarray(adata_train.obsm[k]),
                np.asarray(adata_test.obsm[k]),
            ])

    combined = ad.AnnData(X=X, obs=obs)
    combined.obs_names = list(adata_train.obs_names) + list(adata_test.obs_names)
    for k, arr in obsm.items():
        combined.obsm[k] = arr

    combined.uns["rows"] = {
        "train": slice(0, n_tr),
        "test": slice(n_tr, n_tr + n_te),
    }
    return combined


def _run_harmony_stage2(
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    adata_combined: ad.AnnData,
    stage2_input_key: str,
    stage2_output_key: str,
    max_iter: int = 20,
) -> None:
    """
    Run Harmony stage-2 on combined [train; test] using ``combined_batch``.

    Parameters
    ----------
    adata_train : ad.AnnData
        Training AnnData (modified in-place).
    adata_test : ad.AnnData
        Test AnnData (modified in-place).
    adata_combined : ad.AnnData
        Combined AnnData (modified in-place).
    stage2_input_key : str
        Input obsm key (stage-1 embeddings).
    stage2_output_key : str
        Output obsm key for stage-2 corrected embeddings.
    max_iter : int
        Maximum Harmony iterations.
    """
    Z_tr = np.asarray(adata_train.obsm[stage2_input_key], dtype=np.float32)
    Z_te = np.asarray(adata_test.obsm[stage2_input_key], dtype=np.float32)
    Z_all = np.vstack([Z_tr, Z_te])

    meta = pd.DataFrame({
        "combined_batch": adata_combined.obs["combined_batch"].astype(str).values,
    })

    logger.info(
        "Harmony stage-2: {} total samples, combined_batch groups: {}",
        Z_all.shape[0], meta["combined_batch"].nunique(),
    )

    ho = hm.run_harmony(Z_all, meta_data=meta, vars_use=["combined_batch"], max_iter_harmony=max_iter)
    Zc = np.asarray(ho.Z_corr, dtype=np.float32)
    if Zc.shape[0] != Z_all.shape[0] and Zc.shape[1] == Z_all.shape[0]:
        Zc = Zc.T

    n_tr = Z_tr.shape[0]
    adata_train.obsm[stage2_output_key] = Zc[:n_tr]
    adata_test.obsm[stage2_output_key] = Zc[n_tr:]
    adata_combined.obsm[stage2_output_key] = Zc

    logger.info("Harmony stage-2 complete. Output shape: {}", Zc.shape)


def _qc_split_mixing(
    adata_combined: ad.AnnData,
    rep_key: str,
    split_col: str,
    n_neighbors: int = 30,
    metric: str = "cosine",
) -> dict[str, float]:
    """
    Compute per-cell split-mixing score on combined kNN graph.

    Parameters
    ----------
    adata_combined : ad.AnnData
        Combined AnnData with ``.obsm[rep_key]``.
    rep_key : str
        Embedding key for neighbor computation.
    split_col : str
        Column distinguishing train/test.
    n_neighbors : int
        Number of neighbors.
    metric : str
        Distance metric.

    Returns
    -------
    dict[str, float]
        Mean mixing overall and by split.
    """
    sc.pp.neighbors(adata_combined, use_rep=rep_key, n_neighbors=n_neighbors, metric=metric)

    G = adata_combined.obsp["connectivities"].tocsr()
    split = adata_combined.obs[split_col].to_numpy()
    labels = {lab: i for i, lab in enumerate(np.unique(split))}
    lab_i = np.vectorize(labels.get)(split)

    mix = np.zeros(adata_combined.n_obs, dtype=np.float32)
    indptr, indices = G.indptr, G.indices
    for i in range(adata_combined.n_obs):
        nbrs = indices[indptr[i]:indptr[i + 1]]
        mix[i] = np.nan if nbrs.size == 0 else float((lab_i[nbrs] != lab_i[i]).mean())
    adata_combined.obs["qc_mix_score"] = mix

    return {
        "mean_mix_overall": float(np.nanmean(mix)),
        "mean_mix_by_split": (
            adata_combined.obs.groupby(split_col)["qc_mix_score"].mean().to_dict()
        ),
    }


def _compute_ood_mask(
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    stage2_key: str,
    metric: str = "cosine",
    ref_k: int = 2,
    q: float = 0.995,
    factor: float = 1.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Flag test cells as OOD based on 2nd-NN distance in stage-2 space.

    Parameters
    ----------
    adata_train : ad.AnnData
        Training AnnData with ``.obsm[stage2_key]``.
    adata_test : ad.AnnData
        Test AnnData with ``.obsm[stage2_key]``.
    stage2_key : str
        Embedding key in ``.obsm``.
    metric : str
        Distance metric.
    ref_k : int
        k for reference distribution (2nd-NN among train).
    q : float
        Quantile for threshold.
    factor : float
        Multiplier for threshold.

    Returns
    -------
    tuple[np.ndarray, dict[str, float]]
        Boolean OOD mask and diagnostic statistics.
    """
    Ztr = np.asarray(adata_train.obsm[stage2_key], dtype=np.float32)
    Zte = np.asarray(adata_test.obsm[stage2_key], dtype=np.float32)

    nn_tr = NearestNeighbors(n_neighbors=ref_k, metric=metric, algorithm="brute").fit(Ztr)
    d_tr, _ = nn_tr.kneighbors(Ztr, return_distance=True)
    ref_dist = d_tr[:, 1].astype(np.float32)
    thr = float(np.quantile(ref_dist, q) * factor)

    nn_te = NearestNeighbors(n_neighbors=1, metric=metric, algorithm="brute").fit(Ztr)
    d_te, _ = nn_te.kneighbors(Zte, return_distance=True)
    d_te = d_te[:, 0].astype(np.float32)

    ood = d_te > thr
    stats = {
        "threshold": thr,
        "q": float(q),
        "factor": float(factor),
        "pct_ood": float(100.0 * ood.mean()),
    }
    logger.debug("OOD detection: {:.2f}% flagged (thr={:.4f})", stats["pct_ood"], thr)
    return ood, stats


def _project_test_to_train_umap(
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    adata_combined: ad.AnnData,
    stage2_key: str,
    umap_train_key: str,
    k_fallback: int = 30,
    metric: str = "cosine",
    ood_mask: np.ndarray | None = None,
    ood_mode: str = "stage2_knn",
) -> dict[str, float | int]:
    """
    Project test cells onto frozen train UMAP via connectivity barycenter.

    Parameters
    ----------
    adata_train : ad.AnnData
        Training AnnData with frozen UMAP in ``.obsm[umap_train_key]``.
    adata_test : ad.AnnData
        Test AnnData (modified in-place with projected UMAP).
    adata_combined : ad.AnnData
        Combined AnnData with connectivities.
    stage2_key : str
        Stage-2 embedding key for fallback kNN.
    umap_train_key : str
        UMAP key in train ``.obsm``.
    k_fallback : int
        k for gKNN fallback projection.
    metric : str
        Distance metric for fallback.
    ood_mask : np.ndarray or None
        Boolean mask for OOD test cells.
    ood_mode : str
        How to handle OOD cells: ``'downweight'``, ``'skip'``, ``'stage2_knn'``.

    Returns
    -------
    dict[str, float | int]
        Projection diagnostics.
    """
    Utr = np.asarray(adata_train.obsm[umap_train_key], dtype=np.float32)
    C = adata_combined.obsp["connectivities"].tocsr()

    n_tr = adata_train.n_obs
    n_te = adata_test.n_obs
    rows_test = np.arange(n_tr, n_tr + n_te)
    cols_train = np.arange(0, n_tr)

    C_sub = C[rows_test, :][:, cols_train].tocsr()

    # apply OOD handling
    if ood_mask is not None:
        eff_ood = np.asarray(ood_mask, dtype=bool)
        if ood_mode == "downweight":
            D = sparse.diags(np.where(eff_ood, 0.5, 1.0).astype(np.float32))
            C_sub = D @ C_sub
        elif ood_mode in ("skip", "stage2_knn"):
            z = np.where(eff_ood)[0]
            if len(z):
                C_sub[z, :] = 0.0
    else:
        eff_ood = None

    # base projection
    rs = np.array(C_sub.sum(axis=1)).ravel().astype(np.float32)
    has_links = rs > 0
    C_norm = C_sub.copy()
    if has_links.any():
        C_norm[has_links] = C_norm[has_links].multiply(1.0 / (rs[has_links][:, None] + EPS))
    Ute = np.asarray(C_norm @ Utr, dtype=np.float32)

    # fallback for rows with no links
    fallback_rows = np.where(~has_links)[0]
    if fallback_rows.size:
        Ute = _gknn_fallback(adata_train, adata_test, stage2_key, metric, k_fallback, Utr, Ute, fallback_rows)

    # stage2_knn override for OOD
    if eff_ood is not None and ood_mode == "stage2_knn":
        ood_rows = np.where(eff_ood)[0]
        if ood_rows.size:
            Ute = _gknn_fallback(adata_train, adata_test, stage2_key, metric, k_fallback, Utr, Ute, ood_rows)

    adata_test.obsm[umap_train_key] = Ute

    return {
        "n_test": int(n_te),
        "n_rows_no_train_links": int((~has_links).sum()),
        "pct_rows_no_train_links": float(100.0 * (~has_links).mean()),
        "ood_mode": ood_mode,
    }


def _gknn_fallback(
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    stage2_key: str,
    metric: str,
    k: int,
    Utr: np.ndarray,
    Ute: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    """
    Gaussian-weighted kNN fallback for UMAP projection.

    Parameters
    ----------
    adata_train : ad.AnnData
        Training AnnData.
    adata_test : ad.AnnData
        Test AnnData.
    stage2_key : str
        Embedding key in ``.obsm``.
    metric : str
        Distance metric.
    k : int
        Number of neighbors.
    Utr : np.ndarray
        Train UMAP coordinates.
    Ute : np.ndarray
        Test UMAP coordinates (modified in-place).
    rows : np.ndarray
        Row indices needing fallback.

    Returns
    -------
    np.ndarray
        Updated test UMAP coordinates.
    """
    Ztr = np.asarray(adata_train.obsm[stage2_key], dtype=np.float32)
    Zte = np.asarray(adata_test.obsm[stage2_key], dtype=np.float32)
    kk = min(k, Ztr.shape[0])

    nn = NearestNeighbors(n_neighbors=kk, metric=metric, algorithm="brute").fit(Ztr)
    d, idx = nn.kneighbors(Zte[rows], return_distance=True)
    sig = np.median(d, axis=1, keepdims=True) + EPS
    W = np.exp(-(d ** 2) / (2.0 * sig ** 2)).astype(np.float64)
    W /= W.sum(axis=1, keepdims=True) + EPS
    Ute[rows] = (W[..., None] * Utr[idx]).sum(axis=1).astype(np.float32)
    return Ute


def barycentric_stage1_embeddings(
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    stage1_key: str,
    stage2_key: str,
    k: int = 30,
    metric: str = "cosine",
    adaptive_sigma: bool = True,
    write_key: str | None = None,
) -> dict[str, int | str | bool]:
    """
    Align test stage-1 embeddings to train via Gaussian barycentric projection.

    For each test cell, find k nearest train neighbors in stage-2 space,
    then compute weighted average of their stage-1 embeddings.

    Parameters
    ----------
    adata_train : ad.AnnData
        Training AnnData.
    adata_test : ad.AnnData
        Test AnnData (modified in-place).
    stage1_key : str
        Stage-1 embedding key in ``.obsm``.
    stage2_key : str
        Stage-2 embedding key in ``.obsm``.
    k : int
        Number of neighbors in stage-2.
    metric : str
        Distance metric for neighbor search.
    adaptive_sigma : bool
        Use per-row median distance as sigma.
    write_key : str or None
        Output key. None overwrites ``stage1_key`` in test.

    Returns
    -------
    dict[str, int | str | bool]
        Summary statistics.
    """
    Ztr = np.asarray(adata_train.obsm[stage2_key], dtype=np.float32)
    Zte = np.asarray(adata_test.obsm[stage2_key], dtype=np.float32)
    Etr = np.asarray(adata_train.obsm[stage1_key], dtype=np.float32)

    kk = min(max(1, k), Ztr.shape[0])
    nn = NearestNeighbors(n_neighbors=kk, metric=metric, algorithm="brute").fit(Ztr)
    d, idx = nn.kneighbors(Zte, return_distance=True)

    if adaptive_sigma:
        sig = np.median(d, axis=1, keepdims=True) + EPS
    else:
        sig = np.mean(d, axis=1, keepdims=True) + EPS

    W = np.exp(-(d ** 2) / (2.0 * sig ** 2)).astype(np.float64)
    W /= W.sum(axis=1, keepdims=True) + EPS

    Ete_bar = (W[..., None] * Etr[idx]).sum(axis=1).astype(np.float32)

    out_key = stage1_key if write_key is None else write_key
    adata_test.obsm[out_key] = Ete_bar

    logger.info("Barycentric stage-1 projection: {} test cells -> '{}'", Ete_bar.shape[0], out_key)
    return {"k": kk, "metric": metric, "adaptive_sigma": adaptive_sigma, "write_key": out_key}


def robustness_index(
    adata: ad.AnnData,
    emb_key: str = "X_umap",
    bio_key: str = "diagnosis",
    conf_key: str = "combined_batch",
    k: int = 50,
    metric: str = "euclidean",
) -> tuple[float, dict[str, int | float]]:
    """
    Compute Robustness Index: RI = SO / (SO + OS).

    SO = same biology, different confounder among k-NN.
    OS = other biology, same confounder among k-NN.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with embedding, bio labels, and confounder labels.
    emb_key : str
        Key in ``.obsm`` for the embedding to evaluate.
    bio_key : str
        Column in ``.obs`` for biological labels (e.g. diagnosis).
    conf_key : str
        Column in ``.obs`` for confounder labels (e.g. combined_batch).
    k : int
        Number of neighbors.
    metric : str
        Distance metric.

    Returns
    -------
    tuple[float, dict[str, int | float]]
        RI value and detail dict with SO, OS counts.
    """
    X = np.asarray(adata.obsm[emb_key])
    y_bio = adata.obs[bio_key].to_numpy()
    y_conf = adata.obs[conf_key].to_numpy()

    valid = ~(pd.isna(y_bio) | pd.isna(y_conf))
    X, y_bio, y_conf = X[valid], y_bio[valid], y_conf[valid]
    n = X.shape[0]

    kk = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=kk, metric=metric, algorithm="brute").fit(X)
    neigh_idx = nn.kneighbors(return_distance=False)[:, 1:]

    same_bio = y_bio[neigh_idx] == y_bio[:, None]
    same_conf = y_conf[neigh_idx] == y_conf[:, None]

    SO = int((same_bio & ~same_conf).sum())
    OS = int((~same_bio & same_conf).sum())
    denom = SO + OS
    ri = (SO / denom) if denom > 0 else np.nan

    logger.info("Robustness Index: {:.3f} (SO={}, OS={})", ri, SO, OS)
    return ri, {"SO": SO, "OS": OS, "n_used": n, "k": k}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def backtrack_harmony_integration(
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    *,
    embedding_key: str,
    batch_col: str = "batch",
    split_col: str = "split",
    umap_train_key: str = "UMAP_Harmony_",
    stage2_output_key: str = "Embeddings_Harmony_Stage_2",
    qc_neighbors: int = 30,
    qc_metric: str = "cosine",
    ood_q: float = 0.995,
    ood_factor: float = 1.0,
    ood_mode: str = "stage2_knn",
    k_fallback: int = 30,
    max_iter_stage2: int = 20,
) -> tuple[ad.AnnData, dict]:
    """
    Full BackTrack Harmony Integration pipeline.

    Steps:
      1. Build combined AnnData with ``combined_batch``.
      2. Harmony stage-2 on combined (train + test).
      3. QC: split-mixing score.
      4. OOD detection in stage-2 space.
      5. Project test onto frozen train UMAP.
      6. Barycentric stage-1 embedding correction.

    Parameters
    ----------
    adata_train : ad.AnnData
        Training AnnData with stage-1 Harmony embeddings and frozen UMAP.
    adata_test : ad.AnnData
        Test AnnData with raw stage-1 embeddings.
    embedding_key : str
        Raw embedding key in ``.obsm`` (stage-1 input).
    batch_col : str
        Batch column in ``.obs``.
    split_col : str
        Split column in ``.obs``.
    umap_train_key : str
        Key for the frozen train UMAP in ``.obsm``.
    stage2_output_key : str
        Output key for stage-2 corrected embeddings.
    qc_neighbors : int
        Number of neighbors for QC mixing.
    qc_metric : str
        Metric for neighbor computation.
    ood_q : float
        Quantile for OOD threshold.
    ood_factor : float
        Multiplier for OOD threshold.
    ood_mode : str
        OOD handling mode: ``'downweight'``, ``'skip'``, ``'stage2_knn'``.
    k_fallback : int
        k for gKNN fallback.
    max_iter_stage2 : int
        Max Harmony iterations for stage-2.

    Returns
    -------
    tuple[ad.AnnData, dict]
        Combined AnnData and diagnostic dict with qc, ood, projection stats.
    """
    logger.info("[1/6] Building combined AnnData with combined_batch")
    combined = _make_combined(
        adata_train, adata_test,
        batch_col=batch_col, split_col=split_col,
        obsm_keys=(embedding_key, stage2_output_key, umap_train_key),
    )
    logger.info(
        "    combined: {} obs (train={}, test={})",
        combined.n_obs, adata_train.n_obs, adata_test.n_obs,
    )

    logger.info("[2/6] Harmony stage-2 on combined_batch")
    _run_harmony_stage2(
        adata_train, adata_test, combined,
        stage2_input_key=embedding_key,
        stage2_output_key=stage2_output_key,
        max_iter=max_iter_stage2,
    )

    logger.info("[3/6] QC: split-mixing score")
    qc_stats = _qc_split_mixing(
        combined, rep_key=stage2_output_key, split_col=split_col,
        n_neighbors=qc_neighbors, metric=qc_metric,
    )
    logger.info("    mean mix overall: {:.3f}", qc_stats["mean_mix_overall"])

    logger.info("[4/6] OOD detection in stage-2")
    ood_mask, ood_stats = _compute_ood_mask(
        adata_train, adata_test,
        stage2_key=stage2_output_key, metric=qc_metric,
        ref_k=2, q=ood_q, factor=ood_factor,
    )
    adata_test.obs["stage2_ood"] = ood_mask
    logger.info("    OOD: {:.2f}%", ood_stats["pct_ood"])

    logger.info("[5/6] Projecting test -> frozen train UMAP (ood_mode='{}')", ood_mode)
    proj_stats = _project_test_to_train_umap(
        adata_train, adata_test, combined,
        stage2_key=stage2_output_key,
        umap_train_key=umap_train_key,
        k_fallback=k_fallback, metric=qc_metric,
        ood_mask=ood_mask, ood_mode=ood_mode,
    )

    logger.info("[6/6] Barycentric stage-1 correction")
    barycentric_stage1_embeddings(
        adata_train, adata_test,
        stage1_key=embedding_key,
        stage2_key=stage2_output_key,
        k=k_fallback, metric=qc_metric,
    )

    # final stacked UMAP for plotting
    U_train = np.asarray(adata_train.obsm[umap_train_key], dtype=np.float32)
    U_test = np.asarray(adata_test.obsm[umap_train_key], dtype=np.float32)
    combined.obsm["X_umap"] = np.vstack([U_train, U_test])

    logger.info("BackTrack Harmony Integration complete.")
    diag = {"qc": qc_stats, "ood": ood_stats, "projection": proj_stats}
    return combined, diag
