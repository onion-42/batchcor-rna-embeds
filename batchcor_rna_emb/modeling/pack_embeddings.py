"""Pack precomputed scGPT embeddings into AnnData objects.

This script does not run scGPT inference. It loads raw .zarr AnnData files from
`data/raw/`, loads precomputed `.npy` embeddings from `embeddings/`, and stores
embeddings, PCA, and UMAP projections directly into `adata.obsm`. The updated
AnnData objects are saved as `.h5ad` files in `data/processed/`.
"""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from loguru import logger
from sklearn.decomposition import PCA
import umap


logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SCGPT_METADATA = {
    "model_version": "scGPT-human",
    "environment": "batcor_env",
}


def cohort_name_from_zarr(path: Path) -> str:
    name = path.name
    if name.endswith(".adata.zarr"):
        return name[: -len(".adata.zarr")]
    if name.endswith(".zarr"):
        return name[: -len(".zarr")]
    return name


def find_raw_cohorts() -> list[Path]:
    zarr_paths = sorted(RAW_DIR.glob("*.zarr"))
    if not zarr_paths:
        logger.error("No .zarr files found in %s", RAW_DIR)
    return zarr_paths


def _decode_bytes_array(array: np.ndarray) -> np.ndarray:
    if array.dtype.kind == "S":
        return array.astype(str)
    return array


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
                    fallback = np.zeros_like(codes, dtype=np.int64)
                    return pd.Categorical.from_codes(fallback, categories)

                if codes.min() >= 1 and (codes.max() - codes.min()) < categories.size:
                    shifted = (codes - codes.min()).astype(np.int64)
                    try:
                        return pd.Categorical.from_codes(shifted, categories)
                    except ValueError:
                        pass

                fixed_labels = categories[np.asarray(codes, dtype=np.int64) % categories.size]
                return pd.Categorical(fixed_labels, categories=categories)
        if "data" in element and hasattr(element["data"], "shape"):
            return _decode_bytes_array(np.asarray(element["data"][:]))

    raise ValueError(f"Unsupported zarr element type for manual loading: {type(element)}")


def _load_zarr_dataframe(group: zarr.hierarchy.Group) -> pd.DataFrame:
    data = {}
    index = None
    if "_index" in group:
        index = _load_zarr_element(group["_index"])

    for key in group:
        if key == "_index":
            continue
        data[key] = _load_zarr_element(group[key])

    if index is None:
        return pd.DataFrame(data)
    return pd.DataFrame(data, index=pd.Index(index))


def load_anndata_from_zarr(raw_path: Path) -> ad.AnnData:
    try:
        return ad.read_zarr(str(raw_path))
    except Exception as exc:
        logger.warning(f"anndata.read_zarr failed: {exc}. Falling back to manual zarr loader.")

    root = zarr.open_group(str(raw_path), mode="r")
    if "X" not in root:
        raise ValueError(f"Missing 'X' matrix in zarr dataset: {raw_path}")

    x = np.asarray(root["X"][:])
    obs = _load_zarr_dataframe(root["obs"])
    var = _load_zarr_dataframe(root["var"])

    return ad.AnnData(X=x, obs=obs, var=var)


def pack_cohort(raw_path: Path) -> None:
    cohort_name = cohort_name_from_zarr(raw_path)
    npy_path = EMBEDDINGS_DIR / f"{cohort_name}_scgpt_embeddings.npy"

    logger.info(f"Processing cohort: {cohort_name}")
    logger.info(f"  raw zarr: {raw_path}")
    logger.info(f"  embeddings: {npy_path}")

    if not raw_path.exists():
        logger.error(f"Raw .zarr dataset not found: {raw_path}")
        return

    if not npy_path.exists():
        logger.error(f"Embedding file not found for cohort '{cohort_name}': {npy_path}")
        return

    adata = load_anndata_from_zarr(raw_path)
    embeddings = np.load(str(npy_path))

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2D, but got shape {embeddings.shape} for {npy_path}"
        )

    if adata.n_obs != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch between AnnData observations ({adata.n_obs}) and embeddings rows ({embeddings.shape[0]}) for {cohort_name}"
        )

    embeddings = embeddings.astype(np.float32)
    adata.obsm["scGPT_embedding"] = embeddings
    logger.info(f"  Added scGPT_embedding {embeddings.shape}")

    logger.info("  Computing PCA (128D) on embeddings")
    pca = PCA(n_components=128, random_state=0)
    pca_embeds = pca.fit_transform(embeddings).astype(np.float32)
    adata.obsm["PCA128d_scGPT_embedding"] = pca_embeds
    logger.info(f"  Added PCA128d_scGPT_embedding {pca_embeds.shape}")

    logger.info("  Computing UMAP (3D) on embeddings")
    umap_model = umap.UMAP(n_components=3, random_state=0, metric="euclidean")
    umap_embeds = umap_model.fit_transform(embeddings).astype(np.float32)
    adata.obsm["UMAP3d_scGPT_embedding"] = umap_embeds
    logger.info(f"  Added UMAP3d_scGPT_embedding {umap_embeds.shape}")

    adata.uns["scGPT_metadata"] = SCGPT_METADATA
    logger.info("  Added scGPT_metadata")

    out_path = PROCESSED_DIR / f"{cohort_name}.h5ad"
    adata.write_h5ad(str(out_path))
    logger.info(f"  Saved processed AnnData to {out_path}")


def main() -> int:
    logger.info("Starting embedding packaging script")
    raw_cohorts = find_raw_cohorts()
    if not raw_cohorts:
        return 1

    for raw_path in raw_cohorts:
        try:
            pack_cohort(raw_path)
        except Exception as exc:
            logger.exception(f"Failed to pack cohort {raw_path.name}: {exc}")
            return 1

    logger.info(f"Finished packaging embeddings for {len(raw_cohorts)} cohorts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
