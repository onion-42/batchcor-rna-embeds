"""Загрузка и сохранение AnnData из/в Zarr формат."""
from __future__ import annotations

from pathlib import Path

import anndata as ad
from loguru import logger


def load_cohort(path: str | Path) -> ad.AnnData:
    """
    Load a single AnnData cohort from a Zarr store.

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

    adata = ad.read_zarr(path)
    logger.info(
        "Loaded cohort '{}': {} samples x {} genes",
        path.stem, adata.n_obs, adata.n_vars,
    )
    return adata


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
