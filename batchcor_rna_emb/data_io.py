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
    """Read Zarr store with maximum cross-version compatibility.

    Tries ``ad.read_zarr`` first with duplicate-category patching.
    On ``IORegistryError`` (anndata/zarr version mismatch), falls back
    to manual zarr reading that bypasses anndata's IOSpec registry.

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
        return adata
    except Exception as e:
        logger.warning("ad.read_zarr failed ({}), using manual fallback", e)
    finally:
        pd.Categorical.from_codes = _original_from_codes

    # ── Fallback: read zarr store manually ──
    return _read_zarr_manual(path)


def _read_zarr_manual(path: Path) -> ad.AnnData:
    """Read zarr v2 store manually, bypassing anndata's IOSpec registry."""
    import json

    import numpy as np
    import zarr

    logger.info("Manual zarr read: {}", path)
    root = zarr.open(str(path), mode="r")

    # --- X matrix ---
    X = None
    if "X" in root:
        x_elem = root["X"]
        if isinstance(x_elem, zarr.Array):
            X = np.array(x_elem)
        elif hasattr(x_elem, "keys"):
            # Sparse CSR/CSC stored as group with data/indices/indptr
            import scipy.sparse as sp
            data = np.array(x_elem["data"])
            indices = np.array(x_elem["indices"])
            indptr = np.array(x_elem["indptr"])
            attrs = dict(x_elem.attrs) if hasattr(x_elem, "attrs") else {}
            enc = attrs.get("encoding-type", "csr_matrix")
            shape = tuple(attrs.get("shape", []))
            if "csc" in enc:
                X = sp.csc_matrix((data, indices, indptr), shape=shape)
            else:
                X = sp.csr_matrix((data, indices, indptr), shape=shape)

    # --- obs / var DataFrames ---
    obs = _read_dataframe_from_zarr(root, "obs")
    var = _read_dataframe_from_zarr(root, "var")

    # --- obsm / varm ---
    obsm = {}
    if "obsm" in root:
        for key in root["obsm"].keys():
            elem = root["obsm"][key]
            if isinstance(elem, zarr.Array):
                obsm[key] = np.array(elem)

    varm = {}
    if "varm" in root:
        for key in root["varm"].keys():
            elem = root["varm"][key]
            if isinstance(elem, zarr.Array):
                varm[key] = np.array(elem)

    # --- uns (best-effort) ---
    uns = {}
    if "uns" in root:
        for key in root["uns"].keys():
            try:
                elem = root["uns"][key]
                if isinstance(elem, zarr.Array):
                    val = np.array(elem)
                    uns[key] = val.item() if val.ndim == 0 else val
            except Exception:
                pass

    adata = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm, varm=varm, uns=uns)
    logger.info("Manual read OK: {} x {}", adata.n_obs, adata.n_vars)
    return adata


def _read_dataframe_from_zarr(root, frame_name: str) -> pd.DataFrame:
    """Read obs or var DataFrame from zarr store manually."""
    import json

    import numpy as np
    import zarr

    if frame_name not in root:
        return pd.DataFrame()

    frame = root[frame_name]
    attrs = dict(frame.attrs) if hasattr(frame, "attrs") else {}

    # Get column order from _index and column-order attrs
    index_col = attrs.get("_index", "_index")
    col_order = attrs.get("column-order", [])

    data = {}
    index = None

    for col_name in frame.keys():
        elem = frame[col_name]

        if hasattr(elem, "keys") and "codes" in elem and "categories" in elem:
            # Categorical column
            codes = np.array(elem["codes"])
            cats = np.array(elem["categories"])
            # Deduplicate
            cat_list = list(cats)
            if len(cat_list) != len(set(cat_list)):
                unique = list(dict.fromkeys(cat_list))
                old_to_new = {i: unique.index(c) for i, c in enumerate(cat_list)}
                codes = np.array([old_to_new.get(int(c), c) if c >= 0 else c for c in codes], dtype=np.intp)
                cat_list = unique
            ordered = bool(elem.attrs.get("ordered", False)) if hasattr(elem, "attrs") else False
            cat_type = pd.CategoricalDtype(categories=cat_list, ordered=ordered)
            series = pd.Categorical.from_codes(codes, dtype=cat_type)
            data[col_name] = series
        elif isinstance(elem, zarr.Array):
            arr = np.array(elem)
            # Check for null encoding
            elem_attrs = dict(elem.attrs) if hasattr(elem, "attrs") else {}
            enc_type = elem_attrs.get("encoding-type", "")
            if enc_type == "null":
                continue  # Skip null-encoded arrays
            data[col_name] = arr
        else:
            continue

        if col_name == index_col:
            index = data.pop(col_name)

    # Build DataFrame with column order
    if col_order:
        ordered_data = {}
        for c in col_order:
            if c in data:
                ordered_data[c] = data[c]
        # Add any remaining
        for c in data:
            if c not in ordered_data:
                ordered_data[c] = data[c]
        data = ordered_data

    df = pd.DataFrame(data)
    if index is not None:
        if hasattr(index, "tolist"):
            df.index = pd.Index(index)
        else:
            df.index = pd.Index(list(index))

    return df


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
    import numpy as np
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sanitize object columns to prevent Zarr string array errors
    # (e.g., 'expected unicode string, found 3.0')
    for df in [adata.obs, adata.var]:
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).replace("nan", np.nan).astype("category")
                
    adata.write_zarr(path)
    logger.info("Saved AnnData ({} x {}) to '{}'", adata.n_obs, adata.n_vars, path)
