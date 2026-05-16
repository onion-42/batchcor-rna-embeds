"""
v5_strict_pipeline.py
=====================
Curator-compliant evaluation on ``OS_bin_35months`` with strict leakage controls.

Architectural guarantees
------------------------
1. **No target leakage** — ``LEAKY_COLS`` never enter feature matrix ``X``.
2. **No embedding scaling** — PCA on raw / cAE-corrected scGPT without
   ``StandardScaler`` on the manifold.
3. **Clinical scaling only** — ``StandardScaler`` fits on numeric MFP / Kassandra /
   auxiliary clinical columns only.
4. **cAE inside CV** — ``ConditionalAutoencoder`` is trained on each training fold
   only, then applied to train + validation folds (neutral decoder vectors).
5. **Diagnosis-conditioned batches** — batch one-hot uses ``Diagnosis`` (not
   ``Cohort``) so correction mixes batches within the same disease.

Workflow
--------
  * Load ``data/processed/UNIFIED_Cohort.h5ad`` (or build path via env).
  * Restrict to ``obs['split'] == 'train'`` for stratified 5-fold CV.
  * Per fold: fit PCA → fit cAE on raw scGPT → correct → build features → LightGBM.
  * Log mean ± std ROC-AUC across folds.
  * Optionally score held-out ``split == 'test'`` rows (external sanity check).

Usage::

    python -m batchcor_rna_emb.stress_test.v5_strict_pipeline

Environment
-----------
* ``V5_SEED`` — RNG seed (default 42).
* ``V5_SMOKE=1`` — 2-fold CV, fewer cAE epochs (fast dev).
* ``V5_UNIFIED`` — path to unified h5ad (default ``data/processed/UNIFIED_Cohort.h5ad``).
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import lightgbm as lgb
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from batchcor_rna_emb.batch_correction.cae import (
    CAEConfig,
    ConditionalAutoencoder,
    correct_embeddings,
    train_cae,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | <cyan>{function}</cyan> | {message}"
    ),
    level=os.environ.get("LOG_LEVEL", "INFO"),
)


# =============================================================================
# CONFIG
# =============================================================================

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("{}={!r} invalid; using {}", name, raw, default)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "y"}


SEED: int = _env_int("V5_SEED", 42)
SMOKE: bool = _env_bool("V5_SMOKE", False)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
UNIFIED_PATH = Path(
    os.environ.get("V5_UNIFIED", str(PROCESSED_DIR / "UNIFIED_Cohort.h5ad"))
)

METRICS_DIR = REPO_ROOT / "metrics_csv"
V5_CV_CSV = METRICS_DIR / "v5_os_bin35_cv_results.csv"
V5_OOD_CSV = METRICS_DIR / "v5_os_bin35_ood_per_cohort.csv"

SCGPT_KEY = "scGPT_embedding"
CAE_KEY = "cAE_embedding_v5_fold"  # per-fold CV (in-memory)
CAE_KEY_FINAL = "cAE_embedding_v5_final"  # global train → OOD scoring
TARGET_COL = "OS_bin_35months"
DIAGNOSIS_COL = "Diagnosis"
COHORT_COL = "cohort"

N_SPLITS: int = 2 if SMOKE else 5
PCA_DIM: int = 128
N_FOLDS = N_SPLITS

# Must NEVER appear in X (targets / survival / response proxies).
LEAKY_COLS: tuple[str, ...] = (
    "PFS",
    "OS",
    "PFS_FLAG",
    "OS_FLAG",
    "Response",
    "RECIST",
    "Recist",
    "Responder",
    "BOR",
    "Benefit",
    "OS_bin_35months",
    "PFS_bin",
    "pfs_stratificator",
    "os_stratificator",
    "PFS_Response",
)

# Clinical numeric blocks (scaled); never includes LEAKY_COLS.
MFP_PREFIX = "MFP_"
KASSANDRA_PREFIX = "Kassandra_"
NUMERIC_CLINICAL_EXTRA: tuple[str, ...] = ("Age", "TMB", "PDL1_TC_IHC_num")

# Safe categoricals for one-hot (Diagnosis allowed — drives cAE, also a feature).
CATEGORICAL_SAFE: tuple[tuple[str, str], ...] = (
    (DIAGNOSIS_COL, "Diag"),
    ("Therapy_group", "Tg"),
    ("Pat_Condition_MSKCC", "MSKCC"),
    ("Stage", "Stage"),
    ("Gender", "Gen"),
)

# cAE hyper-parameters (lighter than global run_cae_correction for CV speed)
CAE_LATENT_DIM = 160
CAE_MAX_EPOCHS = 80 if SMOKE else 250
CAE_PATIENCE = 15 if SMOKE else 25
CAE_BATCH_SIZE = 128
CAE_LR = 8e-4
CAE_WEIGHT_DECAY = 5e-4


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# TARGET & COHORT HELPERS
# =============================================================================

def extract_binary_target(obs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (y, mask) for ``OS_bin_35months``.

    y is float {0,1}; mask True where label is valid.
    """
    if TARGET_COL not in obs.columns:
        raise RuntimeError(
            f"Missing {TARGET_COL} in obs. Run harmonize_targets.py and "
            "build_unified_adata.py first."
        )
    y = pd.to_numeric(obs[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(y) & np.isin(y, (0.0, 1.0))
    return y.astype(np.int64), mask


def _select_clinical_numeric(obs: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in obs.columns:
        if c in LEAKY_COLS:
            continue
        if c.startswith(MFP_PREFIX) or c.startswith(KASSANDRA_PREFIX):
            cols.append(c)
    for c in NUMERIC_CLINICAL_EXTRA:
        if c in obs.columns and c not in LEAKY_COLS:
            cols.append(c)
    return sorted(set(cols))


def _build_categorical_block(
    obs: pd.DataFrame,
    train_categories: dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    blocks: list[np.ndarray] = []
    names: list[str] = []
    cat_dict: dict[str, list[str]] = {}

    for col, prefix in CATEGORICAL_SAFE:
        if col in LEAKY_COLS:
            continue
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
        names.extend(f"{prefix}_{c}" for c in cats)

    if not blocks:
        return np.zeros((len(obs), 0), dtype=np.float32), [], cat_dict
    return np.concatenate(blocks, axis=1), names, cat_dict


@dataclass
class FeatureFitState:
    pca: PCA | None
    clin_scaler: StandardScaler | None
    clin_cols: list[str]
    extra_num: list[str]
    clin_medians: dict[str, float]
    cat_dict: dict[str, list[str]]
    feat_names: list[str]


def build_features(
    adata: ad.AnnData,
    embedding_key: str,
    pca_dim: int = PCA_DIM,
    fitted: FeatureFitState | None = None,
) -> tuple[np.ndarray, list[str], FeatureFitState]:
    """
    Build feature matrix **without** target leakage or embedding scaling.

    * scGPT (or cAE-corrected) block: PCA only — **no** StandardScaler.
    * Clinical / Kassandra / MFP block: median impute + StandardScaler only.
    """
    emb = np.asarray(adata.obsm[embedding_key], dtype=np.float32)
    n = emb.shape[0]

    if pca_dim is not None and pca_dim < emb.shape[1]:
        if fitted is None:
            pca = PCA(n_components=pca_dim, random_state=SEED)
            emb_red = pca.fit_transform(emb).astype(np.float32)
        else:
            pca = fitted.pca
            assert pca is not None
            emb_red = pca.transform(emb).astype(np.float32)
        emb_names = [f"embPC{i + 1}" for i in range(emb_red.shape[1])]
    else:
        pca = None
        emb_red = emb
        emb_names = [f"emb_{i}" for i in range(emb.shape[1])]

    obs = adata.obs

    if fitted is None:
        clin_cols = _select_clinical_numeric(obs)
        extra_num = [c for c in NUMERIC_CLINICAL_EXTRA if c in clin_cols]
        clin_cols = [c for c in clin_cols if c not in extra_num]
    else:
        clin_cols = list(fitted.clin_cols)
        extra_num = list(fitted.extra_num)

    all_num = clin_cols + extra_num
    num_df = pd.DataFrame(index=obs.index)
    for c in all_num:
        if c in obs.columns:
            num_df[c] = pd.to_numeric(obs[c], errors="coerce").astype(np.float32)
        else:
            num_df[c] = np.float32("nan")

    miss_cols = [f"miss_{c}" for c in extra_num]
    miss_arr = np.zeros((n, len(extra_num)), dtype=np.float32)
    for j, c in enumerate(extra_num):
        miss_arr[:, j] = num_df[c].isna().astype(np.float32).values

    if fitted is None:
        clin_medians = num_df.median(numeric_only=True).fillna(0.0).to_dict()
    else:
        clin_medians = fitted.clin_medians
    num_df = num_df.fillna(value=clin_medians).fillna(0.0)
    clin_arr = num_df.values.astype(np.float32)

    if fitted is None:
        cat_arr, cat_names, cat_dict = _build_categorical_block(obs, None)
    else:
        cat_arr, cat_names, cat_dict = _build_categorical_block(
            obs, train_categories=fitted.cat_dict
        )

    if fitted is None:
        clin_scaler = StandardScaler().fit(clin_arr) if clin_arr.shape[1] else None
    else:
        clin_scaler = fitted.clin_scaler

    if clin_scaler is not None and clin_arr.shape[1]:
        clin_scaled = clin_scaler.transform(clin_arr).astype(np.float32)
    else:
        clin_scaled = clin_arr

    X = np.concatenate([emb_red, clin_scaled, miss_arr, cat_arr], axis=1)
    feat_names = emb_names + all_num + miss_cols + cat_names

    state = FeatureFitState(
        pca=pca,
        clin_scaler=clin_scaler,
        clin_cols=clin_cols,
        extra_num=extra_num,
        clin_medians=clin_medians,
        cat_dict=cat_dict,
        feat_names=feat_names,
    )
    return X, feat_names, state


# =============================================================================
# DIAGNOSIS-SPECIFIC cAE (per CV fold)
# =============================================================================

def _diagnosis_as_str(diag: pd.Series) -> pd.Series:
    """Coerce ``Diagnosis`` to string (safe for Categorical after concat)."""
    return pd.Series(diag, dtype=object).fillna("Unknown").astype(str)


def _encode_diagnosis_batches(
    diag_train: pd.Series,
    diag_val: pd.Series,
    le: LabelEncoder | None = None,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, int]:
    """
    Map ``Diagnosis`` strings → contiguous batch indices for the cAE.

    Unseen diagnoses in the validation fold are mapped to index 0 (fallback).
    """
    diag_train = _diagnosis_as_str(diag_train)
    diag_val = _diagnosis_as_str(diag_val)

    if le is None:
        le = LabelEncoder()
        le.fit(sorted(diag_train.unique()))
    classes = list(le.classes_)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    idx_train = np.array([class_to_idx.get(d, 0) for d in diag_train], dtype=np.int64)
    idx_val = np.array([class_to_idx.get(d, 0) for d in diag_val], dtype=np.int64)
    return idx_train, idx_val, le, len(classes)


def _encode_diagnosis_with_le(diag: pd.Series, le: LabelEncoder) -> np.ndarray:
    """Map diagnoses using a LabelEncoder fitted on TRAIN (OOD fallback → 0)."""
    diag = _diagnosis_as_str(diag)
    class_to_idx = {c: i for i, c in enumerate(le.classes_)}
    return np.array([class_to_idx.get(d, 0) for d in diag], dtype=np.int64)


def _cae_config(emb_dim: int, n_batches: int) -> CAEConfig:
    h1 = max(256, int(emb_dim * 0.75))
    h2 = max(128, int(emb_dim * 0.50))
    return CAEConfig(
        emb_dim=emb_dim,
        n_batches=n_batches,
        latent_dim=CAE_LATENT_DIM,
        hidden_dims=[h1, h2],
        lr=CAE_LR,
        weight_decay=CAE_WEIGHT_DECAY,
        batch_size=CAE_BATCH_SIZE,
        max_epochs=CAE_MAX_EPOCHS,
        patience=CAE_PATIENCE,
        seed=SEED,
        normalize_emb=True,
    )


def fit_correct_fold(
    adata_tr: ad.AnnData,
    adata_va: ad.AnnData,
) -> tuple[ad.AnnData, ad.AnnData]:
    """
    Train diagnosis-conditioned cAE on ``adata_tr``; correct both splits.

    Uses raw ``scGPT_embedding`` as input; writes corrected vectors to
    ``obsm['cAE_embedding_v5_fold']`` on returned copies.
    """
    if SCGPT_KEY not in adata_tr.obsm:
        raise KeyError(f"Missing obsm['{SCGPT_KEY}'] on training fold")

    emb_tr = np.asarray(adata_tr.obsm[SCGPT_KEY], dtype=np.float32)
    emb_va = np.asarray(adata_va.obsm[SCGPT_KEY], dtype=np.float32)
    emb_dim = emb_tr.shape[1]

    if DIAGNOSIS_COL not in adata_tr.obs.columns:
        raise KeyError(f"Missing obs['{DIAGNOSIS_COL}'] — required for diagnosis-specific cAE")

    batch_tr, batch_va, _, n_batches = _encode_diagnosis_batches(
        adata_tr.obs[DIAGNOSIS_COL],
        adata_va.obs[DIAGNOSIS_COL],
    )
    if n_batches < 1:
        raise RuntimeError("No diagnosis labels for cAE conditioning")

    cfg = _cae_config(emb_dim, n_batches)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "  cAE | train_n={} val_n={} | emb_dim={} | n_diagnoses={} | device={}",
        adata_tr.n_obs,
        adata_va.n_obs,
        emb_dim,
        n_batches,
        device,
    )

    model = train_cae(emb_tr, batch_tr, cfg=cfg, device=device)
    corr_tr = correct_embeddings(model, emb_tr, batch_tr, device=device)
    corr_va = correct_embeddings(model, emb_va, batch_va, device=device)

    out_tr = adata_tr.copy()
    out_va = adata_va.copy()
    out_tr.obsm[CAE_KEY] = corr_tr
    out_va.obsm[CAE_KEY] = corr_va
    return out_tr, out_va


def fit_cae_global_train(
    adata_train: ad.AnnData,
) -> tuple[ConditionalAutoencoder, LabelEncoder]:
    """
    Fit diagnosis-conditioned cAE on the **entire** TRAIN split (all patients).

    Decoder uses neutral (zero) batch vectors at inference via ``correct_embeddings``.
    """
    if SCGPT_KEY not in adata_train.obsm:
        raise KeyError(f"Missing obsm['{SCGPT_KEY}'] on train split")
    if DIAGNOSIS_COL not in adata_train.obs.columns:
        raise KeyError(f"Missing obs['{DIAGNOSIS_COL}'] — required for diagnosis-specific cAE")

    emb = np.asarray(adata_train.obsm[SCGPT_KEY], dtype=np.float32)
    batch_idx, _, diag_le, n_batches = _encode_diagnosis_batches(
        adata_train.obs[DIAGNOSIS_COL],
        adata_train.obs[DIAGNOSIS_COL],
    )
    cfg = _cae_config(emb.shape[1], n_batches)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Global cAE | train_n={} | emb_dim={} | n_diagnoses={} | device={}",
        adata_train.n_obs,
        emb.shape[1],
        n_batches,
        device,
    )
    model = train_cae(emb, batch_idx, cfg=cfg, device=device)
    return model, diag_le


def apply_cae_correction(
    model: ConditionalAutoencoder,
    diag_le: LabelEncoder,
    adata: ad.AnnData,
    *,
    obsm_key: str = CAE_KEY_FINAL,
) -> ad.AnnData:
    """Apply trained cAE with neutral decoder; write corrected embeddings to ``obsm``."""
    emb = np.asarray(adata.obsm[SCGPT_KEY], dtype=np.float32)
    batch_idx = _encode_diagnosis_with_le(adata.obs[DIAGNOSIS_COL], diag_le)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corrected = correct_embeddings(model, emb, batch_idx, device=device)
    out = adata.copy()
    out.obsm[obsm_key] = corrected
    return out


def _make_lgbm_classifier(*, n_estimators: int = 500) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


# =============================================================================
# CV LOOP
# =============================================================================

def run_stratified_cv(adata_train: ad.AnnData) -> pd.DataFrame:
    """
    5-fold stratified CV on ``OS_bin_35months`` with in-fold cAE + LightGBM.
    """
    y_all, mask = extract_binary_target(adata_train.obs)
    if mask.sum() < N_FOLDS * 4:
        raise RuntimeError(
            f"Too few labeled patients for {N_FOLDS}-fold CV: {mask.sum()} valid rows"
        )

    idx_labeled = np.where(mask)[0]
    y = y_all[mask]
    X_dummy = np.zeros(len(y))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows: list[dict] = []
    fold_aucs: list[float] = []

    for fold, (tr_rel, va_rel) in enumerate(skf.split(X_dummy, y), start=1):
        tr_idx = idx_labeled[tr_rel]
        va_idx = idx_labeled[va_rel]
        logger.info(
            "Fold {}/{} | train={} val={} | class balance train={:.2f} pos",
            fold,
            N_FOLDS,
            len(tr_idx),
            len(va_idx),
            y[tr_rel].mean(),
        )

        adata_tr = adata_train[tr_idx].copy()
        adata_va = adata_train[va_idx].copy()

        # (c)–(d) Diagnosis-specific cAE inside the fold
        adata_tr, adata_va = fit_correct_fold(adata_tr, adata_va)

        # (b)+(e) Features: PCA without embedding scale; clinical scaler fit on train
        X_tr, feat_names, fit_state = build_features(
            adata_tr, embedding_key=CAE_KEY, pca_dim=PCA_DIM, fitted=None
        )
        X_va, _, _ = build_features(
            adata_va, embedding_key=CAE_KEY, pca_dim=PCA_DIM, fitted=fit_state
        )

        y_tr = y_all[tr_idx]
        y_va = y_all[va_idx]

        clf = lgb.LGBMClassifier(
            n_estimators=400 if not SMOKE else 80,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
        clf.fit(X_tr, y_tr)

        proba = clf.predict_proba(X_va)[:, 1]
        if len(np.unique(y_va)) < 2:
            auc = float("nan")
            logger.warning("  Fold {}: single class in val — AUC undefined", fold)
        else:
            auc = float(roc_auc_score(y_va, proba))
            fold_aucs.append(auc)
            logger.info("  Fold {} ROC-AUC = {:.4f}", fold, auc)

        rows.append({
            "fold": fold,
            "n_train": len(tr_idx),
            "n_val": len(va_idx),
            "roc_auc": auc,
            "n_features": len(feat_names),
            "pca_dim": PCA_DIM,
            "model": "LightGBM",
            "target": TARGET_COL,
            "embedding_in": CAE_KEY,
            "cae_conditioning": DIAGNOSIS_COL,
        })

    df = pd.DataFrame(rows)
    valid = df["roc_auc"].dropna()
    mean_auc = float(valid.mean()) if len(valid) else float("nan")
    std_auc = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
    logger.success(
        "CV complete | mean ROC-AUC = {:.4f} ± {:.4f} ({} folds)",
        mean_auc,
        std_auc,
        len(valid),
    )
    df.attrs["mean_roc_auc"] = mean_auc
    df.attrs["std_roc_auc"] = std_auc
    return df


def evaluate_per_cohort_ood(
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
) -> pd.DataFrame:
    """
    Train final LightGBM on all labeled TRAIN rows after global cAE correction.

    Score each TEST cohort separately (no pooled OOD). Per-cohort patients are
    cAE-corrected with neutral decoder conditioning before feature extraction.
    """
    if COHORT_COL not in adata_test.obs.columns:
        raise KeyError(f"Missing obs['{COHORT_COL}'] on test split")

    y_tr_all, m_tr = extract_binary_target(adata_train.obs)
    if m_tr.sum() < 10 or len(np.unique(y_tr_all[m_tr])) < 2:
        logger.warning("Skipping OOD — insufficient labeled train rows")
        return pd.DataFrame()

    logger.info("=== Per-cohort OOD (OS_bin_35months) ===")
    cae_model, diag_le = fit_cae_global_train(adata_train)
    adata_train_corr = apply_cae_correction(cae_model, diag_le, adata_train)

    tr_idx = np.where(m_tr)[0]
    ad_tr_lab = adata_train_corr[tr_idx]
    X_tr, _, fit_state = build_features(
        ad_tr_lab, CAE_KEY_FINAL, PCA_DIM, fitted=None
    )
    y_tr = y_tr_all[tr_idx]

    clf = _make_lgbm_classifier()
    clf.fit(X_tr, y_tr)
    logger.info(
        "Final classifier | train_labeled={} | n_features={}",
        len(tr_idx),
        X_tr.shape[1],
    )

    cohorts = sorted(adata_test.obs[COHORT_COL].astype(str).unique())
    rows: list[dict] = []

    for cohort in cohorts:
        te_mask = adata_test.obs[COHORT_COL].astype(str) == cohort
        ad_te = adata_test[te_mask].copy()
        y_te_all, m_te = extract_binary_target(ad_te.obs)
        n_labeled = int(m_te.sum())
        n_total = ad_te.n_obs

        if n_labeled < 10 or len(np.unique(y_te_all[m_te])) < 2:
            logger.warning(
                "OOD | {} | skipped (n_labeled={} or single class)",
                cohort,
                n_labeled,
            )
            rows.append({
                "cohort": cohort,
                "split": "test",
                "n_total": n_total,
                "n_labeled": n_labeled,
                "roc_auc": float("nan"),
                "eval_type": "ood_per_cohort",
                "target": TARGET_COL,
                "embedding_in": CAE_KEY_FINAL,
                "cae_conditioning": DIAGNOSIS_COL,
            })
            continue

        ad_te_corr = apply_cae_correction(cae_model, diag_le, ad_te)
        te_idx = np.where(m_te)[0]
        ad_te_lab = ad_te_corr[te_idx]
        X_te, _, _ = build_features(
            ad_te_lab, CAE_KEY_FINAL, PCA_DIM, fitted=fit_state
        )
        proba = clf.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te_all[te_idx], proba))
        logger.info(
            "OOD | {} | n_labeled={}/{} | ROC-AUC = {:.4f}",
            cohort,
            n_labeled,
            n_total,
            auc,
        )
        rows.append({
            "cohort": cohort,
            "split": "test",
            "n_total": n_total,
            "n_labeled": n_labeled,
            "roc_auc": auc,
            "eval_type": "ood_per_cohort",
            "target": TARGET_COL,
            "embedding_in": CAE_KEY_FINAL,
            "cae_conditioning": DIAGNOSIS_COL,
        })

    df = pd.DataFrame(rows)
    valid = df["roc_auc"].dropna()
    if len(valid):
        logger.success(
            "OOD summary | {} cohorts scored | mean AUC = {:.4f} (unweighted)",
            len(valid),
            float(valid.mean()),
        )
    return df


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    set_seed(SEED)
    if SMOKE:
        logger.warning("V5_SMOKE=1 — reduced folds / cAE epochs for quick iteration")

    if not UNIFIED_PATH.is_file():
        logger.error(
            "Unified AnnData not found: {}\n"
            "Run build_unified_adata.py after all cohort h5ads exist.",
            UNIFIED_PATH,
        )
        return 1

    logger.info("Loading {}", UNIFIED_PATH)
    adata = sc.read_h5ad(str(UNIFIED_PATH))

    if "split" not in adata.obs.columns:
        raise RuntimeError("obs['split'] missing — rebuild UNIFIED_Cohort.h5ad")

    train_mask = adata.obs["split"].astype(str) == "train"
    test_mask = adata.obs["split"].astype(str) == "test"
    adata_train = adata[train_mask].copy()
    adata_test = adata[test_mask].copy()

    logger.info(
        "Unified loaded | total={} train={} test={}",
        adata.n_obs,
        adata_train.n_obs,
        adata_test.n_obs,
    )

    cv_df = run_stratified_cv(adata_train)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{
        "mean_roc_auc": cv_df.attrs.get("mean_roc_auc"),
        "std_roc_auc": cv_df.attrs.get("std_roc_auc"),
        "n_folds": N_FOLDS,
        "n_train_labeled": int(extract_binary_target(adata_train.obs)[1].sum()),
        "smoke": SMOKE,
        "seed": SEED,
    }])
    cv_df.to_csv(V5_CV_CSV, index=False)
    summary.to_csv(METRICS_DIR / "v5_os_bin35_summary.csv", index=False)
    logger.success("Wrote {} and v5_os_bin35_summary.csv", V5_CV_CSV)

    ood_df = evaluate_per_cohort_ood(adata_train, adata_test)
    if not ood_df.empty:
        ood_df.to_csv(V5_OOD_CSV, index=False)
        logger.success("Wrote {}", V5_OOD_CSV)
        print("\n" + "=" * 72)
        print("v5 per-cohort OOD ROC-AUC (OS_bin_35months)")
        print("=" * 72)
        for _, row in ood_df.iterrows():
            auc = row["roc_auc"]
            auc_s = f"{auc:.4f}" if pd.notna(auc) else "N/A (skipped)"
            print(
                f"  {row['cohort']:<45}  "
                f"n={int(row['n_labeled']):>4}/{int(row['n_total']):<4}  AUC={auc_s}"
            )
        print("=" * 72 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
