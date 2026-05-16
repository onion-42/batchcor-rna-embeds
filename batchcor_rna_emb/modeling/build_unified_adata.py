"""
Build a single unified AnnData from per-cohort ``data/processed/*.h5ad`` files.

Adds:
  * ``obs['cohort']``   — source file stem
  * ``obs['split']``    — ``train`` (MSK clinical) vs ``test`` (PUB / external)
  * ``obs['OS_bin_35months']`` — harmonised binary OS landmark (if missing)

Writes ``data/processed/UNIFIED_Cohort.h5ad``.

Run (after all cohort h5ads exist, including SCANB)::

    python -m batchcor_rna_emb.modeling.build_unified_adata

Options::

    python -m batchcor_rna_emb.modeling.build_unified_adata --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger

from batchcor_rna_emb.modeling.cohort_registry import (
    REPO_ROOT,
    TEST_COHORT_DUPLICATE_OF_TRAIN,
    split_for_cohort,
)
from batchcor_rna_emb.modeling.harmonize_targets import (
    TARGET_COL,
    compute_os_bin_35months,
    _pick_columns,
)
from batchcor_rna_emb.modeling.pack_embeddings import load_anndata_from_zarr

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    level="INFO",
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_PATH = PROCESSED_DIR / "UNIFIED_Cohort.h5ad"

# Joint objects built from per-cohort exports — skip to avoid duplicate patients.
EXCLUDE_H5AD_STEMS: frozenset[str] = frozenset({
    "TRAIN_Combined_cAE_Corrected",
    "UNIFIED_Cohort",
}) | TEST_COHORT_DUPLICATE_OF_TRAIN

SCGPT_KEY = "scGPT_embedding"


def discover_processed_h5ads(processed_dir: Path) -> list[Path]:
    paths = sorted(processed_dir.glob("*.h5ad"))
    kept = [p for p in paths if p.stem not in EXCLUDE_H5AD_STEMS]
    skipped = [p.stem for p in paths if p.stem in EXCLUDE_H5AD_STEMS]
    if skipped:
        logger.info("Skipping h5ad stems (duplicates / legacy): {}", ", ".join(sorted(skipped)))
    if not kept:
        logger.warning("No per-cohort .h5ad files under {}", processed_dir)
    return kept


def _remove_train_test_index_duplicates(adata: ad.AnnData) -> ad.AnnData:
    """Drop TEST rows whose ``obs`` index already exists in TRAIN (patient leakage)."""
    train_ids = set(adata.obs.index[adata.obs["split"].astype(str) == "train"])
    test_dup = (adata.obs["split"].astype(str) == "test") & adata.obs.index.isin(train_ids)
    n_dup = int(test_dup.sum())
    if n_dup == 0:
        return adata
    dup_cohorts = adata.obs.loc[test_dup, "cohort"].value_counts().to_dict()
    logger.warning(
        "Removing {} test patients with indices already in train: {}",
        n_dup,
        dup_cohorts,
    )
    return adata[~test_dup].copy()


def _ensure_os_bin_35months(adata: ad.AnnData, cohort: str) -> None:
    """Ensure ``OS_bin_35months`` exists in ``adata.obs`` (compute or copy from raw)."""
    if TARGET_COL in adata.obs.columns:
        col = pd.to_numeric(adata.obs[TARGET_COL], errors="coerce")
        adata.obs[TARGET_COL] = col.astype("float64")
        n_valid = int(col.notna().sum())
        logger.info(
            "  {} already has {} (valid={}/{})",
            cohort,
            TARGET_COL,
            n_valid,
            adata.n_obs,
        )
        return

    raw_zarr = RAW_DIR / f"{cohort}.adata.zarr"
    if raw_zarr.is_dir():
        logger.info("  {} missing {} — computing from {}", cohort, TARGET_COL, raw_zarr.name)
        raw = load_anndata_from_zarr(raw_zarr)
        cols = _pick_columns(raw.obs)
        if cols is None:
            logger.warning("  {}: no OS/PFS columns in raw zarr — {} = NaN", cohort, TARGET_COL)
            adata.obs[TARGET_COL] = np.nan
            return
        t = pd.to_numeric(raw.obs[cols.time_col], errors="coerce").to_numpy()
        e = pd.to_numeric(raw.obs[cols.event_col], errors="coerce").to_numpy()
        labels = compute_os_bin_35months(t, e, time_unit=cols.time_unit)
        if not adata.obs.index.equals(raw.obs.index):
            mapped = pd.Series(labels, index=raw.obs.index)
            adata.obs[TARGET_COL] = mapped.reindex(adata.obs.index).to_numpy()
        else:
            adata.obs[TARGET_COL] = labels
        return

    logger.warning(
        "  {}: no {} and no raw zarr — filling NaN",
        cohort,
        TARGET_COL,
    )
    adata.obs[TARGET_COL] = np.nan


def _sanitize_obs_for_h5ad(adata: ad.AnnData) -> None:
    """Coerce mixed-type ``obs`` columns so ``write_h5ad`` does not fail (e.g. Grade)."""
    for col in adata.obs.columns:
        if col == TARGET_COL:
            continue
        s = adata.obs[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        if isinstance(s.dtype, pd.CategoricalDtype):
            adata.obs[col] = s.astype(str)
        elif s.dtype == object:
            adata.obs[col] = s.apply(
                lambda x: str(x) if pd.notna(x) else np.nan
            )
        else:
            adata.obs[col] = s.astype(str)


def _annotate_cohort(adata: ad.AnnData, cohort: str) -> ad.AnnData:
    out = adata.copy()
    out.obs["cohort"] = cohort
    out.obs["split"] = split_for_cohort(cohort)
    if "Diagnosis" not in out.obs.columns and cohort.startswith("PUB_BRCA"):
        out.obs["Diagnosis"] = "BRCA"
    _ensure_os_bin_35months(out, cohort)
    return out


def load_and_annotate(path: Path) -> ad.AnnData:
    cohort = path.stem
    logger.info("Loading {} (n will log after read)", path.name)
    adata = sc.read_h5ad(str(path))
    if SCGPT_KEY not in adata.obsm:
        logger.warning(
            "  {} has no obsm['{}'] — unified object may be incomplete",
            cohort,
            SCGPT_KEY,
        )
    adata = _annotate_cohort(adata, cohort)
    logger.info(
        "  {} → n={} split={} | {} valid labels",
        cohort,
        adata.n_obs,
        adata.obs["split"].iloc[0],
        int(pd.to_numeric(adata.obs[TARGET_COL], errors="coerce").notna().sum()),
    )
    return adata


def build_unified(
    processed_dir: Path = PROCESSED_DIR,
    output_path: Path = OUTPUT_PATH,
    *,
    dry_run: bool = False,
) -> ad.AnnData | None:
    paths = discover_processed_h5ads(processed_dir)
    if not paths:
        return None

    parts: list[ad.AnnData] = []
    for p in paths:
        parts.append(load_and_annotate(p))

    logger.info("Concatenating {} cohorts with merge='same' …", len(parts))
    unified = ad.concat(
        parts,
        axis=0,
        join="outer",
        merge="same",
        uns_merge="same",
        label="cohort_source",
        index_unique="-",
    )

    unified = _remove_train_test_index_duplicates(unified)

    # Normalise target dtype
    unified.obs[TARGET_COL] = pd.to_numeric(
        unified.obs[TARGET_COL], errors="coerce"
    ).astype("float64")

    logger.info(
        "Unified: {} patients × {} genes | train={} test={}",
        unified.n_obs,
        unified.n_vars,
        int((unified.obs["split"] == "train").sum()),
        int((unified.obs["split"] == "test").sum()),
    )
    logger.info("Cohort breakdown:")
    for cohort, n in unified.obs["cohort"].value_counts().sort_index().items():
        spl = unified.obs.loc[unified.obs["cohort"] == cohort, "split"].iloc[0]
        valid = int(
            pd.to_numeric(
                unified.obs.loc[unified.obs["cohort"] == cohort, TARGET_COL],
                errors="coerce",
            )
            .notna()
            .sum()
        )
        logger.info("  {:<42} n={:>5}  split={:<5}  {} valid={}", cohort, n, spl, TARGET_COL, valid)

    _sanitize_obs_for_h5ad(unified)

    if dry_run:
        logger.info("Dry-run: not writing {}", output_path)
        return unified

    output_path.parent.mkdir(parents=True, exist_ok=True)
    unified.write_h5ad(str(output_path))
    logger.success("Wrote {}", output_path)
    return unified


def main() -> int:
    parser = argparse.ArgumentParser(description="Build UNIFIED_Cohort.h5ad")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory with per-cohort .h5ad files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output unified h5ad path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and concat only; do not write file",
    )
    args = parser.parse_args()

    if not args.processed_dir.is_dir():
        logger.error("Processed directory not found: {}", args.processed_dir)
        return 1

    build_unified(
        processed_dir=args.processed_dir,
        output_path=args.output,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
