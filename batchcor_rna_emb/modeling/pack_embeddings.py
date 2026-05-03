"""Pack precomputed scGPT embeddings into AnnData objects.

Loads raw ``.zarr`` AnnData from ``data/raw/``, loads ``embeddings/*_scgpt_embeddings.npy``,
writes ``obsm`` keys, and saves ``data/processed/*.h5ad``. Does **not** run scGPT inference
or any finetuned scGPT path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from loguru import logger
from sklearn.decomposition import PCA
import umap

from batchcor_rna_emb.batch_correction.config import SEED

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)

# Repo root (…/batchcor_rna_emb/modeling/pack_embeddings.py → parents[2])
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SCGPT_METADATA = {
    "model_version": "scGPT-human",
    "environment": "batcor_env",
}

_PCA_N_COMPONENTS = 128
_UMAP_N_COMPONENTS = 3


def cohort_name_from_zarr(path: Path) -> str:
    """Strip ``.adata.zarr`` / ``.zarr`` suffixes without corrupting in-name substrings."""
    name = path.name
    name = name.removesuffix(".zarr")
    name = name.removesuffix(".adata")
    return name


def find_raw_cohorts() -> list[Path]:
    zarr_paths = sorted(DATA_RAW_DIR.glob("*.zarr"))
    if not zarr_paths:
        logger.error("No .zarr files found in {}", DATA_RAW_DIR)
    return zarr_paths


def _decode_bytes_array(array: np.ndarray) -> np.ndarray:
    return array.astype(str) if array.dtype.kind == "S" else array


def _load_zarr_element(element):
    if hasattr(element, "shape") and not hasattr(element, "keys"):
        return _decode_bytes_array(np.asarray(element[:]))

    if hasattr(element, "keys"):
        if "categories" in element and "codes" in element:
            codes = np.asarray(element["codes"][:])
            categories = _decode_bytes_array(np.asarray(element["categories"][:]))
            try:
                return pd.Categorical.from_codes(codes.astype(np.int64), categories)
            except ValueError:
                if categories.size == 1:
                    return pd.Categorical.from_codes(
                        np.zeros_like(codes, dtype=np.int64), categories
                    )
                if codes.min() >= 1 and (codes.max() - codes.min()) < categories.size:
                    try:
                        return pd.Categorical.from_codes(
                            (codes - codes.min()).astype(np.int64), categories
                        )
                    except ValueError:
                        pass
                fixed = categories[np.asarray(codes, dtype=np.int64) % categories.size]
                return pd.Categorical(fixed, categories=categories)
        if "data" in element and hasattr(element["data"], "shape"):
            return _decode_bytes_array(np.asarray(element["data"][:]))

    raise ValueError(f"Unsupported zarr element type: {type(element)}")


def _load_zarr_dataframe(group: zarr.hierarchy.Group) -> pd.DataFrame:
    data: dict = {}
    index = None
    if "_index" in group:
        index = _load_zarr_element(group["_index"])
    for key in group:
        if key == "_index":
            continue
        data[key] = _load_zarr_element(group[key])
    return (
        pd.DataFrame(data, index=pd.Index(index))
        if index is not None
        else pd.DataFrame(data)
    )


def load_anndata_from_zarr(raw_path: Path) -> ad.AnnData:
    try:
        return ad.read_zarr(str(raw_path))
    except Exception as exc:
        logger.warning(
            "anndata.read_zarr failed: {}. Falling back to manual zarr loader.", exc
        )

    root = zarr.open_group(str(raw_path), mode="r")
    if "X" not in root:
        raise ValueError(f"Missing 'X' matrix in zarr: {raw_path}")

    x = np.asarray(root["X"][:])
    obs = _load_zarr_dataframe(root["obs"])
    var = _load_zarr_dataframe(root["var"])
    return ad.AnnData(X=x, obs=obs, var=var)


def pack_cohort(raw_path: Path) -> None:
    cohort_name = cohort_name_from_zarr(raw_path)
    npy_path = EMBEDDINGS_DIR / f"{cohort_name}_scgpt_embeddings.npy"

    logger.info("Processing cohort: {}", cohort_name)
    logger.info("  raw zarr:    {}", raw_path)
    logger.info("  embeddings:  {}", npy_path)

    if not raw_path.exists():
        logger.error("Raw .zarr not found: {}", raw_path)
        return
    if not npy_path.exists():
        logger.error("Embedding .npy not found: {}", npy_path)
        return

    adata = load_anndata_from_zarr(raw_path)
    embeddings = np.load(str(npy_path))

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2-D, got shape {embeddings.shape} for {npy_path}"
        )
    if adata.n_obs != embeddings.shape[0]:
        raise ValueError(
            f"AnnData n_obs={adata.n_obs} != embeddings rows={embeddings.shape[0]} "
            f"for {cohort_name}"
        )

    embeddings = embeddings.astype(np.float32)
    adata.obsm["scGPT_embedding"] = embeddings
    logger.info("  Added scGPT_embedding {}", embeddings.shape)

    emb_dim = embeddings.shape[1]
    pca_n = min(_PCA_N_COMPONENTS, max(1, adata.n_obs - 1), emb_dim)
    if pca_n < _PCA_N_COMPONENTS:
        logger.warning(
            "PCA n_components capped from {} to {} (n_obs={}, emb_dim={}).",
            _PCA_N_COMPONENTS,
            pca_n,
            adata.n_obs,
            emb_dim,
        )
    logger.info("  Computing PCA ({}D) on embeddings", pca_n)
    pca = PCA(n_components=pca_n, random_state=SEED)
    pca_embeds = pca.fit_transform(embeddings).astype(np.float32)
    pca_key = (
        "PCA128d_scGPT_embedding"
        if pca_n == _PCA_N_COMPONENTS
        else f"PCA{pca_n}d_scGPT_embedding"
    )
    adata.obsm[pca_key] = pca_embeds
    logger.info("  Added {} {}", pca_key, pca_embeds.shape)

    umap_n = min(_UMAP_N_COMPONENTS, max(1, adata.n_obs - 1))
    n_neighbors = min(15, max(2, adata.n_obs - 1))
    logger.info(
        "  Computing UMAP ({}D, n_neighbors={}) on embeddings …",
        umap_n,
        n_neighbors,
    )
    umap_model = umap.UMAP(
        n_components=umap_n,
        n_neighbors=n_neighbors,
        random_state=SEED,
        metric="euclidean",
    )
    umap_embeds = umap_model.fit_transform(embeddings).astype(np.float32)
    umap_key = (
        "UMAP3d_scGPT_embedding"
        if umap_n == _UMAP_N_COMPONENTS
        else f"UMAP{umap_n}d_scGPT_embedding"
    )
    adata.obsm[umap_key] = umap_embeds
    logger.info("  Added {} {}", umap_key, umap_embeds.shape)

    adata.uns["scGPT_metadata"] = SCGPT_METADATA

    out_path = DATA_PROCESSED_DIR / f"{cohort_name}.h5ad"
    adata.write_h5ad(str(out_path))
    logger.info("  Saved processed AnnData to {}", out_path)


def main() -> int:
    logger.info("Starting embedding packaging script")
    raw_cohorts = find_raw_cohorts()
    if not raw_cohorts:
        return 1

    for raw_path in raw_cohorts:
        try:
            pack_cohort(raw_path)
        except Exception as exc:
            logger.exception("Failed to pack cohort {}: {}", raw_path.name, exc)
            return 1

    logger.info("Finished packaging embeddings for {} cohorts", len(raw_cohorts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
