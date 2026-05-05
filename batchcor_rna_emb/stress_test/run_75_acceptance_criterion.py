"""
metrics/run_75_acceptance_criterion.py
=======================================
75% Acceptance Criterion Evaluation
-------------------------------------
Implements the pairwise win-rate criterion from:

  Gross et al. (2024) "Robust evaluation of deep learning-based
  representation methods for survival and treatment response prediction
  from histopathology." Nature Scientific Reports.

Method
------
Over N=50 independent random splits (StratifiedShuffleSplit, 80/20,
seeds 1…50), we train a LightGBM classifier on EACH split for two
competing feature sets:

  A  →  Raw scGPT-PCA32 + Clinical covariates
  B  →  cAE-corrected-PCA32 + Clinical covariates

For each split we record the test ROC-AUC for both and mark a WIN for B
if auc_cAE > auc_Raw.  The criterion is MET iff win-rate >= 75%.

All feature engineering is byte-for-byte identical to v4_definitive_pipeline
(same FitState, same StandardScaler, same PCA dimension, same clinical
imputation logic, same label binarisation) so the comparison is strictly fair.

Output files
------------
  metrics/acceptance_75_scores.csv          – per-split paired AUC table
  visualizations/acceptance_75_criterion.png – publication-grade figure

Usage
-----
    python -m batchcor_rna_emb.stress_test.metrics.run_75_acceptance_criterion
    # or directly:
    python metrics/run_75_acceptance_criterion.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from loguru import logger
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

import lightgbm as lgb

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# LOGGING
# =============================================================================
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | <cyan>{function}</cyan> | {message}"
    ),
    level="INFO",
)

# =============================================================================
# PATHS — mirrors v4_definitive_pipeline.py exactly
# =============================================================================
_HERE         : Path = Path(__file__).resolve().parent
_PROJ_ROOT    : Path = _HERE.parents[1]   # project root

PROCESSED_DIR : Path = _PROJ_ROOT / "data" / "processed"
METRICS_DIR   : Path = _PROJ_ROOT / "metrics_csv"
VIZ_DIR       : Path = _PROJ_ROOT / "visualizations"

TRAIN_H5AD    : Path = PROCESSED_DIR / "TRAIN_Combined_cAE_Corrected.h5ad"
SCORES_CSV    : Path = METRICS_DIR / "acceptance_75_scores.csv"
FIGURE_PATH   : Path = VIZ_DIR / "acceptance_75_criterion.png"

# =============================================================================
# CONSTANTS — identical to v4_definitive_pipeline.py
# =============================================================================
SEED          : int  = 42
N_ITERATIONS  : int  = 50           # paper recommendation: ≥ 50 splits
TEST_SIZE     : float= 0.20
WIN_THRESHOLD : float= 0.75

SCGPT_KEY     : str  = "scGPT_embedding"
CAE_KEY       : str  = "cAE_embedding"
PCA_DIM       : int  = 32           # matches SURVIVAL_PCA_DIM in v4

# LightGBM params — exact copy from fit_classifiers() in v4_definitive_pipeline
LGBM_PARAMS: dict = dict(
    n_estimators       = 400,
    learning_rate      = 0.05,
    num_leaves         = 31,
    max_depth          = -1,
    min_child_samples  = 10,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    reg_alpha          = 0.1,
    reg_lambda         = 1.0,
    class_weight       = "balanced",
    n_jobs             = -1,
    verbosity          = -1,
)

# Response keyword heuristics — identical to v4_definitive_pipeline
RESPONSE_KEYWORDS = [
    "response", "responder", "respond", "benefit", "bor", "recist",
    "pfs_response", "pfs_flag", "pfs_stratificator",
]
PREFERRED_RESPONSE_PRIORITY = [
    "response", "responder", "recist", "pfs_response",
    "pfs_flag", "benefit",
]
POSITIVE_LABELS = frozenset([
    "r", "cr", "pr", "responder", "response",
    "benefit", "yes", "1", "true", "durable_benefit",
])
NEGATIVE_LABELS = frozenset([
    "nr", "sd", "pd", "non_responder", "nonresponder",
    "no_response", "no_benefit", "no", "0", "false",
    "non_benefit", "progressive",
])

# Clinical feature columns — identical to v4_definitive_pipeline
CATEGORICAL_FEATURES: list[tuple[str, str]] = [
    ("Diagnosis",           "Diag"),
    ("Cohort",              "Cohort"),
    ("Therapy_group",       "Tg"),
    ("Pat_Condition_MSKCC", "MSKCC"),
    ("Stage",               "Stage"),
    ("Gender",              "Gen"),
]
NUMERIC_CLINICAL_EXTRA: list[str] = ["Age", "TMB", "PDL1_TC_IHC_num"]


# =============================================================================
# RESPONSE LABEL DETECTION  (copied verbatim from v4_definitive_pipeline)
# =============================================================================

def detect_response_column(obs: pd.DataFrame) -> str | None:
    """Auto-detect binary response column in .obs."""
    cands = [c for c in obs.columns
             if any(kw in c.lower() for kw in RESPONSE_KEYWORDS)]
    if not cands:
        return None

    def _pri(c: str) -> tuple:
        lc    = c.lower()
        rank  = next(
            (i for i, p in enumerate(PREFERRED_RESPONSE_PRIORITY) if p in lc),
            len(PREFERRED_RESPONSE_PRIORITY) + 1,
        )
        nuniq = int(obs[c].dropna().astype(str).nunique())
        return (rank, 0 if nuniq <= 10 else 1, len(c), c)

    cands.sort(key=_pri)
    return cands[0]


def binarise_labels(s: pd.Series) -> tuple[pd.Series, dict] | None:
    """Map clinical response strings → binary {0, 1}."""
    s    = s.dropna()
    uniq = s.unique()
    mapping: dict = {}
    ok = True

    for v in uniq:
        vs = str(v).strip().lower()
        if vs in POSITIVE_LABELS:
            mapping[v] = 1
        elif vs in NEGATIVE_LABELS:
            mapping[v] = 0
        else:
            ok = False
            break

    if not ok:
        if len(uniq) > 10:
            return None
        le      = LabelEncoder()
        encoded = le.fit_transform(s.astype(str))
        bin_    = pd.Series(encoded, index=s.index, dtype=np.int64)
        mapping = {c: int(i) for i, c in enumerate(le.classes_)}
        return bin_, mapping

    bin_ = s.map(mapping).astype(np.int64)
    if int((bin_ == 1).sum()) == 0 or int((bin_ == 0).sum()) == 0:
        return None
    return bin_, mapping


# =============================================================================
# FEATURE ENGINEERING  (mirrors build_features in v4_definitive_pipeline exactly)
# =============================================================================

def _select_clinical_columns(obs: pd.DataFrame) -> list[str]:
    mfp_cols = sorted(c for c in obs.columns if c.startswith("MFP_"))
    kas_cols = sorted(c for c in obs.columns if c.startswith("Kassandra_"))
    return mfp_cols + kas_cols


def _build_categorical_onehot(
    obs              : pd.DataFrame,
    train_categories : dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    blocks  : list[np.ndarray] = []
    names   : list[str]        = []
    cat_dict: dict[str, list[str]] = {}

    for col, prefix in CATEGORICAL_FEATURES:
        if train_categories is not None:
            cats = train_categories.get(col, [])
            if not cats:
                continue
        else:
            if col not in obs.columns:
                continue
            cats = sorted(obs[col].dropna().astype(str).unique().tolist())
            if not cats:
                continue
        cat_dict[col] = cats
        arr = np.zeros((len(obs), len(cats)), dtype=np.float32)
        if col in obs.columns:
            col_str = obs[col].astype(str).values
            for j, c in enumerate(cats):
                arr[:, j] = (col_str == c).astype(np.float32)
        blocks.append(arr)
        names.extend([f"{prefix}_{c}" for c in cats])

    if blocks:
        return np.concatenate(blocks, axis=1), names, cat_dict
    return np.zeros((len(obs), 0), dtype=np.float32), [], cat_dict


class FitState:
    """Carries all fitted transformers so test fold uses training statistics."""
    __slots__ = (
        "pca", "scaler", "cat_dict",
        "clin_cols", "extra_num", "clin_medians", "feat_names",
    )

    def __init__(
        self,
        pca          : PCA | None,
        scaler       : StandardScaler,
        cat_dict     : dict[str, list[str]],
        clin_cols    : list[str],
        extra_num    : list[str],
        clin_medians : dict,
        feat_names   : list[str],
    ) -> None:
        self.pca          = pca
        self.scaler       = scaler
        self.cat_dict     = cat_dict
        self.clin_cols    = clin_cols
        self.extra_num    = extra_num
        self.clin_medians = clin_medians
        self.feat_names   = feat_names


def build_features(
    adata        : sc.AnnData,
    embedding_key: str,
    pca_dim      : int | None = None,
    fitted       : FitState | None = None,
) -> tuple[np.ndarray, list[str], FitState]:
    """
    Assemble (embedding [+PCA] + MFP + Kassandra + numeric + categorical) → X.

    Byte-for-byte identical logic to v4_definitive_pipeline.build_features().
    When `fitted` is provided the PCA / scaler / medians / categories from
    the training fold are reused without re-fitting (prevents leakage).
    """
    emb = np.asarray(adata.obsm[embedding_key], dtype=np.float32)

    if pca_dim is not None and pca_dim < emb.shape[1]:
        if fitted is None:
            pca = PCA(n_components=pca_dim, random_state=SEED).fit(emb)
        else:
            pca = fitted.pca
        emb_red   = pca.transform(emb).astype(np.float32)
        emb_names = [f"PC{i + 1}" for i in range(pca_dim)]
    else:
        pca       = None
        emb_red   = emb
        emb_names = [f"emb_{i}" for i in range(emb.shape[1])]

    obs = adata.obs

    clin_cols = (
        list(fitted.clin_cols)
        if fitted is not None
        else _select_clinical_columns(obs)
    )
    extra_num = (
        list(fitted.extra_num)
        if fitted is not None
        else [c for c in NUMERIC_CLINICAL_EXTRA if c in obs.columns]
    )
    all_num = clin_cols + extra_num

    num_df = pd.DataFrame(index=obs.index)
    for c in all_num:
        if c in obs.columns:
            num_df[c] = pd.to_numeric(obs[c], errors="coerce").astype(np.float32)
        else:
            num_df[c] = np.float32("nan")

    miss_cols = [f"miss_{c}" for c in extra_num]
    miss_arr  = np.zeros((len(obs), len(extra_num)), dtype=np.float32)
    for j, c in enumerate(extra_num):
        miss_arr[:, j] = num_df[c].isna().astype(np.float32).values

    clin_medians = (
        fitted.clin_medians
        if fitted is not None
        else num_df.median(numeric_only=True).fillna(0.0).to_dict()
    )
    num_df  = num_df.fillna(value=clin_medians).fillna(0.0)
    num_arr = num_df.values.astype(np.float32)

    train_cats = fitted.cat_dict if fitted is not None else None
    cat_arr, cat_names, cat_dict = _build_categorical_onehot(obs, train_cats)

    X_pre      = np.concatenate([emb_red, num_arr, miss_arr, cat_arr], axis=1)
    feat_names = emb_names + all_num + miss_cols + cat_names

    if fitted is None:
        scaler = StandardScaler().fit(X_pre)
    else:
        scaler = fitted.scaler
    X = scaler.transform(X_pre).astype(np.float32)

    state = FitState(
        pca=pca, scaler=scaler, cat_dict=cat_dict,
        clin_cols=clin_cols, extra_num=extra_num,
        clin_medians=clin_medians, feat_names=feat_names,
    )
    return X, feat_names, state


# =============================================================================
# LIGHTGBM TRAINER
# =============================================================================

def _train_lgbm(
    X_tr : np.ndarray,
    y_tr : np.ndarray,
    seed : int,
) -> lgb.LGBMClassifier:
    """
    Instantiate and fit a LightGBM classifier.  Uses the EXACT same params
    as fit_classifiers() in v4_definitive_pipeline so results are comparable.
    """
    clf = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed)
    clf.fit(X_tr, y_tr)
    return clf


def _auc(clf: lgb.LGBMClassifier, X_te: np.ndarray, y_te: np.ndarray) -> float:
    """Compute ROC-AUC; returns NaN on degenerate labels."""
    try:
        prob = clf.predict_proba(X_te)[:, 1]
        return float(roc_auc_score(y_te, prob))
    except ValueError:
        return float("nan")


# =============================================================================
# 75% ACCEPTANCE CRITERION ENGINE
# =============================================================================

def run_acceptance_criterion(
    adata  : sc.AnnData,
    y      : np.ndarray,    # aligned binary labels (same index order as adata)
    n_iter : int  = N_ITERATIONS,
) -> pd.DataFrame:
    """
    Run N independent StratifiedShuffleSplit evaluations and record paired AUC.

    For each split (seed 1 … N):
      1. Split adata patients into 80% train / 20% test.
      2. Build features for RAW scGPT and cAE SEPARATELY on the TRAIN fold
         (PCA + scaler fitted on train, applied to test — no leakage).
      3. Train LightGBM(seed=split_seed) on each feature set.
      4. Evaluate on test fold.

    Returns
    -------
    DataFrame with columns:
        split_seed, auc_raw, auc_cae, delta (=cae-raw), cae_wins (bool)
    """
    rows: list[dict] = []

    for seed in range(1, n_iter + 1):
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=seed
        )
        tr_idx, te_idx = next(sss.split(np.zeros(len(y)), y))

        adata_tr = adata[tr_idx]
        adata_te = adata[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Guard: both folds must have both classes
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            logger.warning(f"  seed={seed:02d} — degenerate split (single class); skipping.")
            continue

        # ── Model A: Raw scGPT + Clinical ──────────────────────────────────
        X_raw_tr, _, state_raw = build_features(
            adata_tr, embedding_key=SCGPT_KEY, pca_dim=PCA_DIM, fitted=None
        )
        X_raw_te, _, _ = build_features(
            adata_te, embedding_key=SCGPT_KEY, pca_dim=PCA_DIM, fitted=state_raw
        )
        clf_raw = _train_lgbm(X_raw_tr, y_tr, seed=seed)
        auc_raw = _auc(clf_raw, X_raw_te, y_te)

        # ── Model B: cAE Corrected + Clinical ──────────────────────────────
        X_cae_tr, _, state_cae = build_features(
            adata_tr, embedding_key=CAE_KEY, pca_dim=PCA_DIM, fitted=None
        )
        X_cae_te, _, _ = build_features(
            adata_te, embedding_key=CAE_KEY, pca_dim=PCA_DIM, fitted=state_cae
        )
        clf_cae = _train_lgbm(X_cae_tr, y_tr, seed=seed)
        auc_cae = _auc(clf_cae, X_cae_te, y_te)

        delta    = auc_cae - auc_raw
        cae_wins = bool(auc_cae > auc_raw)

        rows.append({
            "split_seed": seed,
            "auc_raw"   : round(auc_raw, 6),
            "auc_cae"   : round(auc_cae, 6),
            "delta"     : round(delta,   6),
            "cae_wins"  : cae_wins,
        })
        logger.info(
            f"  split {seed:02d}/{n_iter} | "
            f"AUC_raw={auc_raw:.4f}  AUC_cAE={auc_cae:.4f}  "
            f"Δ={delta:+.4f}  {'✓ cAE wins' if cae_wins else '✗ Raw wins'}"
        )

    return pd.DataFrame(rows)


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def analyse_results(df: pd.DataFrame) -> dict:
    """
    Compute all summary statistics and the 75% criterion verdict.
    Returns a dict of statistics for logging and plotting.
    """
    valid   = df.dropna(subset=["auc_raw", "auc_cae"])
    n_valid = len(valid)

    raw_aucs = valid["auc_raw"].values
    cae_aucs = valid["auc_cae"].values
    deltas   = valid["delta"].values

    wins     = int(valid["cae_wins"].sum())
    win_rate = wins / n_valid if n_valid > 0 else 0.0

    # Wilcoxon signed-rank test (paired, two-sided)
    # zero_method="wilcox" discards ties (delta==0)
    try:
        stat, pval = wilcoxon(
            cae_aucs, raw_aucs,
            alternative="greater",
            zero_method="wilcox",
        )
    except ValueError:
        stat, pval = float("nan"), float("nan")

    return {
        "n_splits"      : n_valid,
        "n_wins_cae"    : wins,
        "win_rate"      : win_rate,
        "criterion_met" : win_rate >= WIN_THRESHOLD,

        "raw_mean"      : float(np.nanmean(raw_aucs)),
        "raw_std"       : float(np.nanstd(raw_aucs)),
        "raw_median"    : float(np.nanmedian(raw_aucs)),

        "cae_mean"      : float(np.nanmean(cae_aucs)),
        "cae_std"       : float(np.nanstd(cae_aucs)),
        "cae_median"    : float(np.nanmedian(cae_aucs)),

        "delta_mean"    : float(np.nanmean(deltas)),
        "delta_std"     : float(np.nanstd(deltas)),
        "delta_median"  : float(np.nanmedian(deltas)),

        "wilcoxon_stat" : float(stat),
        "wilcoxon_pval" : float(pval),
    }


def log_verdict(stats: dict) -> None:
    """Print a highly visible verdict banner to the logger."""
    sep = "=" * 70
    logger.info(sep)
    logger.info("  75% ACCEPTANCE CRITERION — RESULTS")
    logger.info(sep)
    logger.info(
        f"  Splits evaluated    : {stats['n_splits']}"
    )
    logger.info(
        f"  cAE wins            : {stats['n_wins_cae']} / {stats['n_splits']}"
    )
    logger.info(
        f"  Win rate            : {stats['win_rate'] * 100:.1f}%  "
        f"(threshold = {WIN_THRESHOLD * 100:.0f}%)"
    )
    logger.info(
        f"  Raw scGPT AUC       : {stats['raw_mean']:.4f} ± {stats['raw_std']:.4f}"
    )
    logger.info(
        f"  cAE Corrected AUC   : {stats['cae_mean']:.4f} ± {stats['cae_std']:.4f}"
    )
    logger.info(
        f"  Mean Δ (cAE - Raw)  : {stats['delta_mean']:+.4f} ± {stats['delta_std']:.4f}"
    )
    logger.info(
        f"  Wilcoxon p-value    : {stats['wilcoxon_pval']:.4e}  "
        f"(stat = {stats['wilcoxon_stat']:.2f})"
    )
    logger.info(sep)

    if stats["criterion_met"]:
        logger.success(
            "██████████████████████████████████████████████████████████████████\n"
            "  ✅  CRITERION MET  ✅\n"
            f"  cAE batch correction outperforms Raw scGPT in "
            f"{stats['win_rate'] * 100:.1f}% of splits\n"
            f"  (≥ {WIN_THRESHOLD * 100:.0f}% required).  Wilcoxon p = "
            f"{stats['wilcoxon_pval']:.4e}.\n"
            "  The cAE representation is statistically superior for\n"
            "  immunotherapy response prediction across independent splits.\n"
            "██████████████████████████████████████████████████████████████████"
        )
    else:
        logger.warning(
            "██████████████████████████████████████████████████████████████████\n"
            "  ❌  CRITERION NOT MET  ❌\n"
            f"  cAE outperforms Raw scGPT in only "
            f"{stats['win_rate'] * 100:.1f}% of splits\n"
            f"  (< {WIN_THRESHOLD * 100:.0f}% required).  Wilcoxon p = "
            f"{stats['wilcoxon_pval']:.4e}.\n"
            "  Suggested actions:\n"
            "    • Increase cAE latent_dim or reduce patience (less overcorrection)\n"
            "    • Check cAE_embedding key is present and not all-zeros\n"
            "    • Verify response label quality / class imbalance\n"
            "██████████████████████████████████████████████████████████████████"
        )
    logger.info(sep)


# =============================================================================
# PUBLICATION-GRADE VISUALISATION
# =============================================================================

def plot_acceptance_criterion(
    df     : pd.DataFrame,
    stats  : dict,
    out    : Path,
) -> None:
    """
    Two-panel publication figure:

    Panel 1 — Violin + Box + Swarm
        Distribution of 50 AUC values for Raw (blue) vs cAE (teal).
        Includes median line, mean marker, and raw data points.

    Panel 2 — Slopegraph (paired trajectory)
        Each split is a line connecting Raw AUC (left) to cAE AUC (right).
        Green  = cAE wins  (cAE > Raw)
        Red    = Raw wins  (Raw >= cAE)
        Opacity scales with |delta| so decisive splits are more visible.
    """
    valid = df.dropna(subset=["auc_raw", "auc_cae"]).reset_index(drop=True)
    n     = len(valid)

    # ── Colour palette ──────────────────────────────────────────────────────
    COL_RAW      = "#4C72B0"     # muted blue
    COL_CAE      = "#1A9E77"     # teal-green (ColorBrewer Set2)
    COL_WIN      = "#27AE60"     # vivid green
    COL_LOSE     = "#E74C3C"     # vivid red
    COL_BG       = "#F8F9FA"
    COL_GRID     = "#DEE2E6"
    FONT_TITLE   = {"fontsize": 12, "fontweight": "bold", "color": "#2C3E50"}
    FONT_LABEL   = {"fontsize": 10, "color": "#495057"}
    FONT_TICK    = {"labelsize": 9}

    # ── Figure layout ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6.5),
        gridspec_kw={"width_ratios": [1.1, 1.6]},
    )
    fig.patch.set_facecolor(COL_BG)
    for ax in axes:
        ax.set_facecolor(COL_BG)

    # ── Build long-form DataFrame for seaborn ─────────────────────────────
    long_df = pd.concat([
        pd.DataFrame({"Model": "Raw scGPT\n+Clinical", "AUC": valid["auc_raw"]}),
        pd.DataFrame({"Model": "cAE Corrected\n+Clinical", "AUC": valid["auc_cae"]}),
    ], ignore_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL 1: Violin + Box + Swarm
    # ─────────────────────────────────────────────────────────────────────────
    ax1  = axes[0]
    pal  = {"Raw scGPT\n+Clinical": COL_RAW, "cAE Corrected\n+Clinical": COL_CAE}

    # Violin (background density)
    sns.violinplot(
        data=long_df, x="Model", y="AUC",
        palette=pal, inner=None, cut=0.3, linewidth=0,
        alpha=0.35, ax=ax1,
    )
    # Box (IQR + median)
    sns.boxplot(
        data=long_df, x="Model", y="AUC",
        palette=pal, width=0.22, linewidth=1.4,
        fliersize=0, boxprops=dict(alpha=0.85),
        medianprops=dict(color="white", linewidth=2.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        ax=ax1,
    )
    # Raw data points (swarm)
    sns.stripplot(
        data=long_df, x="Model", y="AUC",
        palette=pal, size=4.5, alpha=0.65, jitter=True,
        ax=ax1, zorder=3,
    )
    # Mean markers
    for i, (model, col) in enumerate(pal.items()):
        mean_val = long_df[long_df["Model"] == model]["AUC"].mean()
        ax1.scatter(
            i, mean_val, marker="D", s=55, color="white",
            edgecolors=col, linewidths=1.8, zorder=5,
        )

    # Annotations
    raw_med = valid["auc_raw"].median()
    cae_med = valid["auc_cae"].median()
    for i, (lbl, val, col) in enumerate([
        ("Raw scGPT\n+Clinical", raw_med, COL_RAW),
        ("cAE Corrected\n+Clinical", cae_med, COL_CAE),
    ]):
        ax1.text(
            i, val + 0.004, f"med={val:.3f}",
            ha="center", va="bottom", fontsize=8,
            fontweight="bold", color=col,
        )

    # Win-rate badge in top-right
    wr_txt   = f"{stats['win_rate'] * 100:.0f}% win rate"
    met_col  = COL_WIN if stats["criterion_met"] else COL_LOSE
    ax1.text(
        0.97, 0.97, wr_txt,
        transform=ax1.transAxes, ha="right", va="top",
        fontsize=10, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=met_col, alpha=0.9),
    )

    # Wilcoxon annotation
    p_txt = (
        f"Wilcoxon p = {stats['wilcoxon_pval']:.3e}\n"
        f"Δ = {stats['delta_mean']:+.4f} ± {stats['delta_std']:.4f}"
    )
    ax1.text(
        0.5, 0.03, p_txt,
        transform=ax1.transAxes, ha="center", va="bottom",
        fontsize=8.5, color="#495057",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.75),
    )

    ax1.set_title(
        f"AUC Distribution Across {n} Splits\n(80/20 StratifiedShuffleSplit)",
        **FONT_TITLE,
    )
    ax1.set_xlabel("Feature Set", **FONT_LABEL)
    ax1.set_ylabel("ROC-AUC (Response Prediction)", **FONT_LABEL)
    ax1.tick_params(**FONT_TICK)
    ax1.yaxis.grid(True, color=COL_GRID, linewidth=0.7, linestyle="--")
    ax1.set_axisbelow(True)

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL 2: Slopegraph (paired trajectory)
    # ─────────────────────────────────────────────────────────────────────────
    ax2 = axes[1]

    X_LEFT, X_RIGHT = 0.0, 1.0
    max_delta = valid["delta"].abs().max()
    max_delta = max_delta if max_delta > 0 else 1.0   # avoid div-by-zero

    for _, row in valid.iterrows():
        auc_r    = row["auc_raw"]
        auc_c    = row["auc_cae"]
        wins     = row["cae_wins"]
        alpha    = 0.30 + 0.50 * (abs(row["delta"]) / max_delta)
        lw       = 0.8  + 1.20 * (abs(row["delta"]) / max_delta)
        col      = COL_WIN if wins else COL_LOSE

        ax2.plot(
            [X_LEFT, X_RIGHT], [auc_r, auc_c],
            color=col, alpha=alpha, linewidth=lw, zorder=2,
        )

    # Column scatter points
    jitter_r   = valid["auc_raw"].values  + np.random.default_rng(0).uniform(
        -0.003, 0.003, size=n
    )
    jitter_c   = valid["auc_cae"].values  + np.random.default_rng(1).uniform(
        -0.003, 0.003, size=n
    )
    ax2.scatter(
        np.full(n, X_LEFT),  jitter_r,
        s=22, color=COL_RAW, alpha=0.7, zorder=3, edgecolors="none",
    )
    ax2.scatter(
        np.full(n, X_RIGHT), jitter_c,
        s=22, color=COL_CAE, alpha=0.7, zorder=3, edgecolors="none",
    )

    # Mean markers with error bars
    ax2.errorbar(
        X_LEFT, stats["raw_mean"], yerr=stats["raw_std"],
        fmt="D", color=COL_RAW, markersize=9, linewidth=2.0,
        capsize=5, zorder=5,
    )
    ax2.errorbar(
        X_RIGHT, stats["cae_mean"], yerr=stats["cae_std"],
        fmt="D", color=COL_CAE, markersize=9, linewidth=2.0,
        capsize=5, zorder=5,
    )
    # Mean labels
    for xp, mean_v, col, side in [
        (X_LEFT,  stats["raw_mean"], COL_RAW, "right"),
        (X_RIGHT, stats["cae_mean"], COL_CAE, "left"),
    ]:
        ax2.text(
            xp + (-0.04 if side == "right" else 0.04),
            mean_v,
            f"μ={mean_v:.3f}",
            ha=side, va="center",
            fontsize=8.5, fontweight="bold", color=col,
        )

    # X axis labels
    ax2.set_xticks([X_LEFT, X_RIGHT])
    ax2.set_xticklabels(
        ["Raw scGPT\n+Clinical", "cAE Corrected\n+Clinical"],
        fontsize=9.5, fontweight="bold",
    )
    ax2.set_xlim(-0.25, 1.25)

    # Legend
    win_patch  = mpatches.Patch(color=COL_WIN,  alpha=0.85,
                                label=f"cAE wins ({stats['n_wins_cae']}/{n} splits)")
    lose_patch = mpatches.Patch(color=COL_LOSE, alpha=0.85,
                                label=f"Raw wins ({n - stats['n_wins_cae']}/{n} splits)")
    ax2.legend(
        handles=[win_patch, lose_patch],
        loc="upper center", bbox_to_anchor=(0.5, -0.10),
        ncol=2, fontsize=9, framealpha=0.85,
    )

    verdict_txt  = "✅ CRITERION MET" if stats["criterion_met"] else "❌ CRITERION NOT MET"
    verdict_col  = COL_WIN if stats["criterion_met"] else COL_LOSE
    ax2.text(
        0.5, 0.975, verdict_txt,
        transform=ax2.transAxes, ha="center", va="top",
        fontsize=11, fontweight="bold", color="white",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=verdict_col, alpha=0.92,
        ),
    )

    ax2.set_title(
        f"Paired AUC Trajectory per Split (Slopegraph)\n"
        f"Win Rate = {stats['win_rate'] * 100:.1f}%  "
        f"(threshold {WIN_THRESHOLD * 100:.0f}%)",
        **FONT_TITLE,
    )
    ax2.set_ylabel("ROC-AUC (Response Prediction)", **FONT_LABEL)
    ax2.tick_params(**FONT_TICK)
    ax2.yaxis.grid(True, color=COL_GRID, linewidth=0.7, linestyle="--")
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Shared figure title ──────────────────────────────────────────────────
    fig.suptitle(
        "75% Acceptance Criterion — cAE Corrected vs Raw scGPT Embeddings\n"
        f"LightGBM · PCA-{PCA_DIM} + Clinical Covariates · "
        f"N={n} independent splits",
        fontsize=13, fontweight="bold", color="#2C3E50", y=1.02,
    )

    plt.tight_layout(pad=2.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=160, bbox_inches="tight",
                facecolor=COL_BG, edgecolor="none")
    plt.close(fig)
    logger.info(f"Figure saved → {out}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    wall = time.perf_counter()
    np.random.seed(SEED)

    logger.info("=" * 70)
    logger.info("  75% Acceptance Criterion — Gross et al. (2024)")
    logger.info(f"  Splits: {N_ITERATIONS} | Test size: {TEST_SIZE*100:.0f}%")
    logger.info(f"  Classifier: LightGBM | PCA-{PCA_DIM} + Clinical fusion")
    logger.info("=" * 70)

    # ── Load TRAIN AnnData ──────────────────────────────────────────────────
    if not TRAIN_H5AD.exists():
        raise FileNotFoundError(
            f"TRAIN AnnData not found: {TRAIN_H5AD}\n"
            "Run the cAE correction pipeline first:\n"
            "  python run_cae_correction.py"
        )
    logger.info(f"Loading TRAIN: {TRAIN_H5AD}")
    adata = sc.read_h5ad(str(TRAIN_H5AD))
    logger.info(
        f"  n={adata.n_obs:,} patients | obsm={list(adata.obsm.keys())}"
    )

    # Validate required embedding keys
    for key in [SCGPT_KEY, CAE_KEY]:
        if key not in adata.obsm:
            raise KeyError(
                f"Required .obsm key '{key}' not found.\n"
                f"Available keys: {list(adata.obsm.keys())}"
            )

    # ── Detect and binarise response label ──────────────────────────────────
    resp_col = detect_response_column(adata.obs)
    if resp_col is None:
        raise RuntimeError(
            "No response column detected in adata.obs.\n"
            f"Searched for keywords: {RESPONSE_KEYWORDS}\n"
            f"Available columns: {list(adata.obs.columns)}"
        )
    logger.info(f"Response column detected: '{resp_col}'")

    binarised = binarise_labels(adata.obs[resp_col])
    if binarised is None:
        raise RuntimeError(
            f"Could not binarise column '{resp_col}'. "
            "Verify label values are clinical response strings."
        )
    y_series, mapping = binarised
    logger.info(f"Label mapping: {mapping}")
    logger.info(
        f"Label distribution — Positive (1): {int((y_series == 1).sum()):,} | "
        f"Negative (0): {int((y_series == 0).sum()):,}"
    )

    # Align adata to labelled patients only
    valid_idx   = y_series.index
    adata_clf   = adata[adata.obs.index.isin(valid_idx)].copy()
    y           = y_series.loc[adata_clf.obs.index].to_numpy().astype(np.int64)
    logger.info(
        f"Patients with valid labels: {len(y):,} / {adata.n_obs:,}"
    )

    # ── Run 75% Criterion ───────────────────────────────────────────────────
    logger.info(
        f"Running {N_ITERATIONS} StratifiedShuffleSplit evaluations …"
    )
    scores_df = run_acceptance_criterion(adata_clf, y, n_iter=N_ITERATIONS)

    # ── Statistical analysis ─────────────────────────────────────────────────
    stats = analyse_results(scores_df)
    log_verdict(stats)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(str(SCORES_CSV), index=False)
    logger.info(f"Scores saved → {SCORES_CSV}")

    # Append summary statistics row at the end for convenience
    summary_row = pd.DataFrame([{
        "split_seed": "SUMMARY",
        "auc_raw"   : f"{stats['raw_mean']:.4f} ± {stats['raw_std']:.4f}",
        "auc_cae"   : f"{stats['cae_mean']:.4f} ± {stats['cae_std']:.4f}",
        "delta"     : f"{stats['delta_mean']:+.4f} ± {stats['delta_std']:.4f}",
        "cae_wins"  : (
            f"{stats['n_wins_cae']}/{stats['n_splits']} "
            f"({stats['win_rate']*100:.1f}%) — "
            f"{'MET' if stats['criterion_met'] else 'NOT MET'} | "
            f"Wilcoxon p={stats['wilcoxon_pval']:.4e}"
        ),
    }])
    scores_df_full = pd.concat([scores_df, summary_row], ignore_index=True)
    scores_df_full.to_csv(str(SCORES_CSV), index=False)

    # ── Visualisation ────────────────────────────────────────────────────────
    logger.info("Generating publication figure …")
    plot_acceptance_criterion(scores_df, stats, out=FIGURE_PATH)

    elapsed = time.perf_counter() - wall
    logger.info("=" * 70)
    logger.info(
        f"75% Acceptance Criterion complete in {elapsed:.1f}s "
        f"({elapsed / 60:.1f} min)"
    )
    logger.info(f"  Scores CSV : {SCORES_CSV.resolve()}")
    logger.info(f"  Figure     : {FIGURE_PATH.resolve()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
