"""Загрузка и сохранение AnnData из/в Zarr формат."""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd
from loguru import logger


def _dedup_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate categories in categorical columns.

    Zarr stores written by older anndata/pandas versions can contain
    categorical columns with duplicate categories, which pandas >=2.0
    rejects with ``ValueError: Categorical categories must be unique``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to fix (modified in-place).

    Returns
    -------
    pd.DataFrame
        The same DataFrame with deduplicated categorical categories.
    """
    n_fixed = 0
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            cats = df[col].cat.categories
            if not cats.is_unique:
                df[col] = df[col].astype(str).astype("category")
                n_fixed += 1
    if n_fixed:
        logger.debug(
            "Deduplicated categories in {} categorical column(s)", n_fixed,
        )
    return df


def load_cohort(path: str | Path) -> ad.AnnData:
    """
    Load a single AnnData cohort from a Zarr store.

    Handles pandas >=2.0 incompatibility with duplicate categorical
    categories by temporarily patching ``pd.Categorical.from_codes``.

    Parameters
    ----------
    path : str or Path
        Path to the ``.adata.zarr/`` directory.

    Returns
    -------
    ad.AnnData
        Loaded AnnData object.

    Raises
    ------
    FileNotFoundError
        If the Zarr store does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Zarr store not found: {path}")

    adata = _read_zarr_safe(path)

    # Post-load safety net
    _dedup_categorical_columns(adata.obs)
    _dedup_categorical_columns(adata.var)

    logger.info(
        "Loaded cohort '{}': {} samples x {} genes",
        path.stem, adata.n_obs, adata.n_vars,
    )
    return adata


def _read_zarr_safe(path: Path) -> ad.AnnData:
    """Read Zarr store with automatic duplicate-category handling.

    Temporarily patches ``pd.Categorical.from_codes`` to deduplicate
    categories before construction, then restores the original.

    Parameters
    ----------
    path : Path
        Path to Zarr store.

    Returns
    -------
    ad.AnnData
        Loaded AnnData.
    """
    import numpy as np

    _original_from_codes = pd.Categorical.from_codes

    @classmethod  # type: ignore[misc]
    def _patched_from_codes(cls, codes, categories, ordered=None, dtype=None, validate=True):  # noqa: N805
        cats = pd.Index(categories)
        if not cats.is_unique:
            logger.debug("Deduplicating categories in Categorical.from_codes")
            unique_cats = cats.drop_duplicates()
            # Remap codes to deduplicated indices
            old_to_new = {old: unique_cats.get_loc(cats[old]) for old in range(len(cats))}
            codes = np.array(
                [old_to_new.get(int(c), c) if c >= 0 else c for c in codes],
                dtype=np.intp,
            )
            categories = unique_cats
        return _original_from_codes(codes, categories, ordered=ordered, dtype=dtype)

    try:
        pd.Categorical.from_codes = _patched_from_codes
        adata = ad.read_zarr(path)
    finally:
        pd.Categorical.from_codes = _original_from_codes

    return adata


def _fix_zarr_duplicate_categories(zarr_path: Path) -> None:
    """Fix duplicate categories directly in a Zarr store on disk.

    Walks through ``obs/`` and ``var/`` groups looking for categorical
    arrays (groups with ``categories`` and ``codes`` sub-arrays) and
    deduplicates the ``categories`` array, remapping ``codes`` accordingly.

    Parameters
    ----------
    zarr_path : Path
        Path to the ``.adata.zarr/`` directory.
    """
    import json

    import numpy as np
    import zarr

    root = zarr.open(str(zarr_path), mode="r+")

    for frame_name in ("obs", "var"):
        if frame_name not in root:
            continue
        frame = root[frame_name]
        for col_name in list(frame.keys()):
            col = frame[col_name]
            if not isinstance(col, zarr.Group):
                continue
            if "categories" not in col or "codes" not in col:
                continue

            cats = col["categories"][:]
            if len(cats) == len(set(cats if cats.dtype.kind != "U" else cats.tolist())):
                continue

            logger.info(
                "Fixing duplicate categories: {}/{}", frame_name, col_name,
            )

            # Build unique categories and remap codes
            unique_cats = list(dict.fromkeys(cats.tolist()))
            old_to_new = {old_idx: unique_cats.index(c) for old_idx, c in enumerate(cats.tolist())}

            codes = col["codes"][:]
            new_codes = np.array(
                [old_to_new.get(int(c), c) if c >= 0 else c for c in codes],
                dtype=codes.dtype,
            )

            # Overwrite
            col["categories"] = np.array(unique_cats, dtype=cats.dtype)
            col["codes"] = new_codes

    logger.info("Zarr duplicate-category fix applied to '{}'", zarr_path.name)


def discover_cohorts(data_dir: str | Path, pattern: str = "*.adata.zarr") -> list[Path]:
    """
    Discover all AnnData Zarr stores in a directory.

    Parameters
    ----------
    data_dir : str or Path
        Root data directory to search.
    pattern : str
        Glob pattern to match Zarr stores.

    Returns
    -------
    list[Path]
        Sorted list of discovered Zarr store paths.
    """
    data_dir = Path(data_dir)
    stores = sorted(data_dir.glob(pattern))
    logger.info("Discovered {} cohort(s) in '{}'", len(stores), data_dir)
    return stores


def load_all_cohorts(data_dir: str | Path, pattern: str = "*.adata.zarr") -> list[ad.AnnData]:
    """
    Discover and load all AnnData Zarr cohorts from a directory.

    Parameters
    ----------
    data_dir : str or Path
        Root data directory containing Zarr stores.
    pattern : str
        Glob pattern to match Zarr stores.

    Returns
    -------
    list[ad.AnnData]
        List of loaded AnnData objects.

    Raises
    ------
    FileNotFoundError
        If no Zarr stores are found.
    """
    stores = discover_cohorts(data_dir, pattern)
    if not stores:
        raise FileNotFoundError(f"No Zarr stores matching '{pattern}' in {data_dir}")
    return [load_cohort(s) for s in stores]


def save_adata_zarr(adata: ad.AnnData, path: str | Path) -> None:
    """
    Save AnnData object to a Zarr store.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object to save.
    path : str or Path
        Destination path for the Zarr store.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_zarr(path)
    logger.info("Saved AnnData ({} x {}) to '{}'", adata.n_obs, adata.n_vars, path)
