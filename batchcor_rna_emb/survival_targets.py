"""Модуль для бинаризации выживаемости (OS/PFS) по медиане внутри когорт."""
from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import check_nans_or_infs
from loguru import logger

from batchcor_rna_emb.config import COHORT_COL, SURVIVAL_COLS, TARGET_PREFIX


def compute_median_survival_per_cohort(
    df: pd.DataFrame, time_col: str, event_col: str, cohort_col: str
) -> dict[str, float]:
    """Calculate median survival time per cohort using Kaplan-Meier.

    If median is not reached, falls back to the mean observed time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with survival and cohort data.
    time_col : str
        Column name for survival time.
    event_col : str
        Column name for event indicator (1=event, 0=censored).
    cohort_col : str
        Column name for cohort grouping.

    Returns
    -------
    dict[str, float]
        Mapping from cohort name to its median survival time.
    """
    medians = {}
    
    # Filter valid rows
    valid_mask = df[time_col].notna() & df[event_col].notna() & df[cohort_col].notna()
    df_valid = df[valid_mask]
    
    for cohort, group in df_valid.groupby(cohort_col):
        T = group[time_col].astype(float)
        E = group[event_col].astype(float)
        
        # Check for bad values
        if len(T) == 0 or check_nans_or_infs(T) or check_nans_or_infs(E):
            logger.warning("Invalid survival data for cohort: {}", cohort)
            medians[cohort] = np.nan
            continue
            
        kmf = KaplanMeierFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                kmf.fit(T, event_observed=E)
                median = kmf.median_survival_time_
            except Exception as e:
                logger.warning("KMF failed for cohort {}: {}", cohort, e)
                median = np.inf
                
        if np.isinf(median) or np.isnan(median):
            # Fallback to mean observed time if median is not reached
            fallback = float(T.mean())
            logger.debug(
                "Median survival not reached for cohort '{}'. Using mean time: {:.1f}", 
                cohort, fallback
            )
            medians[cohort] = fallback
        else:
            medians[cohort] = float(median)
            
    return medians


def binarize_survival(
    df: pd.DataFrame, 
    time_col: str, 
    event_col: str, 
    cohort_col: str, 
    medians: dict[str, float]
) -> pd.Series:
    """Binarize survival into 3 classes based on median.

    0 = Events before median (died/progressed early)
    1 = Censored before median (unknown outcome)
    2 = Survived/censored after median (long survivors)

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    time_col : str
        Time column.
    event_col : str
        Event column.
    cohort_col : str
        Cohort column.
    medians : dict[str, float]
        Precomputed median times per cohort.

    Returns
    -------
    pd.Series
        Binarized survival labels (categorical: 0, 1, 2) or NaN.
    """
    result = pd.Series(index=df.index, dtype="float64")
    
    valid_mask = df[time_col].notna() & df[event_col].notna() & df[cohort_col].notna()
    
    for cohort, median_time in medians.items():
        if pd.isna(median_time):
            continue
            
        mask = valid_mask & (df[cohort_col] == cohort)
        T = df.loc[mask, time_col].astype(float)
        E = df.loc[mask, event_col].astype(float)
        
        # 0: event before median
        class_0 = (T <= median_time) & (E == 1)
        # 1: censored before median
        class_1 = (T <= median_time) & (E == 0)
        # 2: survived past median (event or censored)
        class_2 = (T > median_time)
        
        cohort_result = pd.Series(np.nan, index=T.index)
        cohort_result.loc[class_0] = 0.0
        cohort_result.loc[class_1] = 1.0
        cohort_result.loc[class_2] = 2.0
        
        result.update(cohort_result)
        
    return result


def add_survival_targets(adata: ad.AnnData) -> None:
    """Add binarized OS and PFS targets to AnnData in-place.

    Reads 'OS', 'OS.time', 'PFS', 'PFS.time' based on config.SURVIVAL_COLS.
    Creates 'Target_OS_bin' and 'Target_PFS_bin' in adata.obs.
    
    If COHORT_COL is missing, treats the entire adata as a single cohort.

    Parameters
    ----------
    adata : ad.AnnData
        The annotated data matrix.
    """
    obs = adata.obs.copy()
    
    # If Cohort column is missing, create a temporary one
    temp_cohort_col = COHORT_COL
    if temp_cohort_col not in obs.columns:
        temp_cohort_col = "_temp_cohort"
        obs[temp_cohort_col] = "single_cohort"

    targets_added = []

    # Process OS
    os_time = SURVIVAL_COLS.get("os_time")
    os_event = SURVIVAL_COLS.get("os_event")
    
    # Check if SCANB pre-computed column exists
    if "OS_bin_35months" in obs.columns:
        logger.info("Found pre-computed 'OS_bin_35months', mapping to Target_OS_bin")
        # Ensure it's mapped to 0,1,2 properly if needed, assuming it's already 0/1/2
        adata.obs[f"{TARGET_PREFIX}OS_bin"] = obs["OS_bin_35months"]
        targets_added.append(f"{TARGET_PREFIX}OS_bin")
    elif os_time in obs.columns and os_event in obs.columns:
        logger.info("Computing per-cohort median for OS")
        os_medians = compute_median_survival_per_cohort(
            obs, os_time, os_event, temp_cohort_col
        )
        os_bin = binarize_survival(obs, os_time, os_event, temp_cohort_col, os_medians)
        
        col_name = f"{TARGET_PREFIX}OS_bin"
        adata.obs[col_name] = pd.Categorical(os_bin)
        targets_added.append(col_name)
    else:
        logger.debug("OS columns missing, skipping OS binarization")

    # Process PFS
    pfs_time = SURVIVAL_COLS.get("pfs_time")
    pfs_event = SURVIVAL_COLS.get("pfs_event")
    
    if pfs_time in obs.columns and pfs_event in obs.columns:
        logger.info("Computing per-cohort median for PFS")
        pfs_medians = compute_median_survival_per_cohort(
            obs, pfs_time, pfs_event, temp_cohort_col
        )
        pfs_bin = binarize_survival(obs, pfs_time, pfs_event, temp_cohort_col, pfs_medians)
        
        col_name = f"{TARGET_PREFIX}PFS_bin"
        adata.obs[col_name] = pd.Categorical(pfs_bin)
        targets_added.append(col_name)
    else:
        logger.debug("PFS columns missing, skipping PFS binarization")
        
    if targets_added:
        logger.info("Added survival targets: {}", ", ".join(targets_added))
