"""
Harmonize overall-survival targets across all raw Zarr cohorts.

Adds ``obs['OS_bin_35months']`` to every ``data/raw/*.adata.zarr`` store:

  * **1** — patient survived at least 35 months (OS time >= threshold and
    either censored after threshold or died at/after threshold).
  * **0** — death observed before 35 months.
  * **NaN** — insufficient follow-up (e.g. censored before 35 months without
    reaching the landmark).

Time unit is inferred per cohort (days vs months) from column names and the
median of observed times.

Run::

    python -m batchcor_rna_emb.modeling.harmonize_targets
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from loguru import logger

from batchcor_rna_emb.modeling.cohort_registry import split_for_cohort
from batchcor_rna_emb.modeling.pack_embeddings import (
    cohort_name_from_zarr,
    load_anndata_from_zarr,
)

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    level="INFO",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
TARGET_COL = "OS_bin_35months"
LANDMARK_MONTHS = 35.0
DAYS_PER_MONTH = 30.4375
LANDMARK_DAYS = LANDMARK_MONTHS * DAYS_PER_MONTH

TIME_CANDIDATES: tuple[str, ...] = (
    "OS",
    "OS_DAYS",
    "os",
    "os_days",
    "OS_months",
    "os_months",
    "overall_survival",
    "OverallSurvival",
    "PFS",  # fallback when OS missing
    "PFS_DAYS",
    "pfs",
    "pfs_days",
)
EVENT_CANDIDATES: tuple[str, ...] = (
    "OS_FLAG",
    "OS_EVENT",
    "OS_STATUS",
    "os_flag",
    "os_event",
    "os_status",
    "PFS_FLAG",
    "PFS_EVENT",
    "PFS_STATUS",
    "pfs_flag",
    "pfs_event",
    "pfs_status",
)


@dataclass
class SurvivalColumns:
    time_col: str
    event_col: str
    time_unit: str  # "days" | "months"


def _pick_columns(obs: pd.DataFrame) -> SurvivalColumns | None:
    time_col = next((c for c in TIME_CANDIDATES if c in obs.columns), None)
    event_col = next((c for c in EVENT_CANDIDATES if c in obs.columns), None)
    if time_col is None or event_col is None:
        return None

    tl = time_col.lower()
    if "month" in tl:
        unit = "months"
    elif "day" in tl or time_col in ("OS", "PFS", "os", "pfs"):
        # MSK-style OS/PFS columns are documented in days in this project
        unit = "days"
    else:
        t = pd.to_numeric(obs[time_col], errors="coerce")
        med = float(t.median(skipna=True))
        unit = "months" if med < 200 else "days"
    return SurvivalColumns(time_col=time_col, event_col=event_col, time_unit=unit)


def _to_months(time: np.ndarray, unit: str) -> np.ndarray:
    if unit == "months":
        return time.astype(np.float64)
    return time.astype(np.float64) / DAYS_PER_MONTH


def compute_os_bin_35months(
    time: np.ndarray,
    event: np.ndarray,
    *,
    time_unit: str,
) -> np.ndarray:
    """
    Landmark binary target at 35 months.

    event: 1 = event occurred (death), 0 = censored (alive at last follow-up).
    """
    t_mo = _to_months(time, time_unit)
    out = np.full(len(t_mo), np.nan, dtype=np.float64)
    valid = np.isfinite(t_mo) & np.isfinite(event) & (t_mo >= 0)
    t = t_mo[valid]
    e = event[valid].astype(np.float64)

    died_early = (e >= 0.5) & (t < LANDMARK_MONTHS)
    reached_landmark = t >= LANDMARK_MONTHS
    censored_early = (e < 0.5) & (t < LANDMARK_MONTHS)

    idx = np.where(valid)[0]
    out[idx[died_early]] = 0.0
    out[idx[reached_landmark]] = 1.0
    # censored before landmark without death => unknown
    out[idx[censored_early]] = np.nan
    return out


def harmonize_zarr(zarr_path: Path, *, dry_run: bool = False) -> dict:
    cohort = cohort_name_from_zarr(zarr_path)
    adata = load_anndata_from_zarr(zarr_path)
    cols = _pick_columns(adata.obs)
    if cols is None:
        logger.warning(
            "{}: no OS/PFS time+event pair — {} set to all NaN",
            cohort,
            TARGET_COL,
        )
        label = pd.Series(np.nan, index=adata.obs.index, dtype="float64")
        meta = {"time_col": None, "event_col": None, "time_unit": None}
    else:
        t = pd.to_numeric(adata.obs[cols.time_col], errors="coerce").to_numpy()
        e = pd.to_numeric(adata.obs[cols.event_col], errors="coerce").to_numpy()
        arr = compute_os_bin_35months(t, e, time_unit=cols.time_unit)
        label = pd.Series(arr, index=adata.obs.index, name=TARGET_COL)
        meta = {
            "time_col": cols.time_col,
            "event_col": cols.event_col,
            "time_unit": cols.time_unit,
        }

    n0 = int((label == 0).sum())
    n1 = int((label == 1).sum())
    nn = int(label.isna().sum())
    split = split_for_cohort(cohort)
    logger.info(
        "{} [{}] {}: n={} | 0={} 1={} NaN={} | src {} / {} ({})",
        cohort,
        split,
        TARGET_COL,
        adata.n_obs,
        n0,
        n1,
        nn,
        meta.get("time_col"),
        meta.get("event_col"),
        meta.get("time_unit"),
    )

    if dry_run:
        return {"cohort": cohort, "split": split, **meta, "n0": n0, "n1": n1, "nn": nn}

    adata.obs[TARGET_COL] = label
    if "harmonize_targets" not in adata.uns:
        adata.uns["harmonize_targets"] = {}
    adata.uns["harmonize_targets"].update(
        {
            "landmark_months": LANDMARK_MONTHS,
            "landmark_days": LANDMARK_DAYS,
            **meta,
        }
    )
    adata.write_zarr(str(zarr_path))
    logger.info("  wrote {}", zarr_path)
    return {"cohort": cohort, "split": split, **meta, "n0": n0, "n1": n1, "nn": nn}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Add OS_bin_35months to raw zarr cohorts")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not write zarr stores",
    )
    parser.add_argument(
        "--cohort",
        action="append",
        default=None,
        help="Process only these zarr folder names (repeatable)",
    )
    args = parser.parse_args()

    if not DATA_RAW_DIR.is_dir():
        logger.error("Missing {}", DATA_RAW_DIR)
        return 1

    paths = sorted(DATA_RAW_DIR.glob("*.zarr"))
    if args.cohort:
        wanted = {c if c.endswith(".zarr") else f"{c}.zarr" for c in args.cohort}
        paths = [p for p in paths if p.name in wanted]

    if not paths:
        logger.error("No .zarr cohorts under {}", DATA_RAW_DIR)
        return 1

    rows: list[dict] = []
    for zp in paths:
        rows.append(harmonize_zarr(zp, dry_run=args.dry_run))

    summary = pd.DataFrame(rows)
    out_csv = REPO_ROOT / "metrics_csv" / "harmonize_targets_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    logger.success("Summary -> {}", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
