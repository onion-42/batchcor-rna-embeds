"""Leak-free per-fold orchestrator for downstream ML evaluation.

Ensures that PCA, cVAE batch correction, and ML model training
all happen strictly inside each fold — no data leakage.

Architecture
------------
For each outer fold (Split_seed_0..9):
  1. Split data into train / test
  2. (Optional) cVAE: fit on train embeddings → transform both
  3. PCA: fit on train → transform both
  4. ML model: train on train features → predict on test → metrics
  
Transformations are cached per-fold so multiple models can reuse them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from batchcor_rna_emb.config import BATCH_COL, COMPASS_PT_EMBEDDING_KEY, SCVI_LATENT_DIM
from batchcor_rna_emb.modeling.feature_extraction import fit_pca_pipeline, transform_pca_pipeline
from batchcor_rna_emb.modeling.evaluation import (
    evaluate_binary_classifier,
    youden_threshold,
)
from batchcor_rna_emb.modeling.train import predict_proba
from batchcor_rna_emb.split_utils import get_split_masks


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FoldConfig:
    """Configuration for a single fold run."""

    embedding_key: str = COMPASS_PT_EMBEDDING_KEY
    target_col: str = "Response"
    correction_method: str = "none"  # "none" | "cvae_adv2"
    n_pca: int = 128
    cvae_epochs: int = 100
    cvae_latent_dim: int = SCVI_LATENT_DIM
    seed: int = 42
    # Inner CV for hyperparameter tuning (LogReg, LGBM)
    inner_cv_folds: int = 3


# ---------------------------------------------------------------------------
# Per-fold pipeline
# ---------------------------------------------------------------------------

def _apply_correction(
    X_train: np.ndarray,
    X_test: np.ndarray,
    batch_train: np.ndarray,
    batch_test: np.ndarray,
    cfg: FoldConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit batch correction on train, transform both."""
    if cfg.correction_method == "none":
        return X_train, X_test

    if cfg.correction_method == "cvae_adv2":
        from batchcor_rna_emb.batch_correction.scvi_corrector import (
            CVAEAdv2Config,
            CVAEAdv2Corrector,
        )

        n_unique_batches = len(set(batch_train))
        if n_unique_batches < 2:
            logger.warning(
                "Only {} batch(es) in train — skipping cVAE correction",
                n_unique_batches,
            )
            return X_train, X_test

        cvae_cfg = CVAEAdv2Config(
            latent_dim=cfg.cvae_latent_dim,
            n_epochs=cfg.cvae_epochs,
            normalize=True,
            seed=cfg.seed,
        )
        corrector = CVAEAdv2Corrector(cvae_cfg)
        corrector.fit(X_train, batch_train)
        X_train_corr = corrector.transform(X_train, batch_train)
        X_test_corr = corrector.transform(X_test, batch_test)
        return X_train_corr, X_test_corr

    raise ValueError(f"Unknown correction_method: {cfg.correction_method}")


def _apply_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    cfg: FoldConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on train (no scaling for COMPASS), transform both."""
    scaler, pca = fit_pca_pipeline(
        X_train, n_components=cfg.n_pca, seed=cfg.seed, scale=False
    )
    X_train_pca = transform_pca_pipeline(X_train, scaler, pca)
    X_test_pca = transform_pca_pipeline(X_test, scaler, pca)
    return X_train_pca, X_test_pca


def _train_tabpfn(X_train, y_train, X_test, cfg):
    """Train TabPFN (no hyperparameter tuning needed)."""
    from batchcor_rna_emb.modeling.train import train_tabpfn

    model = train_tabpfn(X_train, y_train, seed=cfg.seed)
    probas_train = predict_proba(model, X_train)
    probas_test = predict_proba(model, X_test)
    return probas_train, probas_test


def _train_logreg(X_train, y_train, X_test, cfg):
    """Train LogisticRegression with inner CV for hyperparameter tuning."""
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
    }

    base_model = LogisticRegression(
        max_iter=1000, solver="lbfgs", random_state=cfg.seed
    )
    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=cfg.inner_cv_folds,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    logger.info(
        "LogReg best params: {}, best CV AUC: {:.4f}",
        grid.best_params_,
        grid.best_score_,
    )

    model = grid.best_estimator_
    probas_train = model.predict_proba(X_train)[:, 1]
    probas_test = model.predict_proba(X_test)[:, 1]
    return probas_train, probas_test


def _train_lgbm(X_train, y_train, X_test, cfg):
    """Train LightGBM with inner CV for hyperparameter tuning."""
    from lightgbm import LGBMClassifier

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "num_leaves": [15, 31],
    }

    base_model = LGBMClassifier(
        random_state=cfg.seed, verbose=-1, n_jobs=-1
    )
    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=cfg.inner_cv_folds,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    logger.info(
        "LGBM best params: {}, best CV AUC: {:.4f}",
        grid.best_params_,
        grid.best_score_,
    )

    model = grid.best_estimator_
    probas_train = model.predict_proba(X_train)[:, 1]
    probas_test = model.predict_proba(X_test)[:, 1]
    return probas_train, probas_test


_MODEL_REGISTRY = {
    "tabpfn": _train_tabpfn,
    "logreg": _train_logreg,
    "lgbm": _train_lgbm,
}


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def run_fold(
    adata,
    split_col: str,
    cfg: FoldConfig,
    model_name: str = "tabpfn",
) -> dict:
    """Run a single fold: correction → PCA → ML → metrics.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with raw embeddings, split columns, and targets.
    split_col : str
        Column in ``.obs`` with ``"train"``/``"test"`` labels.
    cfg : FoldConfig
        Pipeline configuration.
    model_name : str
        One of ``"tabpfn"``, ``"logreg"``, ``"lgbm"``.

    Returns
    -------
    dict
        Evaluation metrics for this fold.
    """
    train_mask, test_mask = get_split_masks(adata, split_col)

    X_all = adata.obsm[cfg.embedding_key].astype(np.float32)
    y_all = adata.obs[cfg.target_col].values
    batch_all = adata.obs[BATCH_COL].values

    # Filter out missing targets and enforce STRICTLY BINARY classes
    valid = pd.notna(y_all) & (y_all.astype(str) != "nan") & (y_all.astype(str) != "Missing")
    valid_vals = y_all[valid]
    if len(valid_vals) > 0:
        top_2_classes = pd.Series(valid_vals).value_counts().nlargest(2).index.values
        valid = valid & np.isin(y_all, top_2_classes)
        
    train_valid = train_mask & valid
    test_valid = test_mask & valid

    X_train = X_all[train_valid]
    X_test = X_all[test_valid]
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(y_all[valid].astype(str))
    y_train = le.transform(y_all[train_valid].astype(str)).astype(float)
    y_test = le.transform(y_all[test_valid].astype(str)).astype(float)
    batch_train = batch_all[train_valid]
    batch_test = batch_all[test_valid]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        logger.warning(
            "Fold '{}': <2 classes in train or test (train classes: {}, test classes: {}). Skipping.",
            split_col, np.unique(y_train), np.unique(y_test),
        )
        return {}

    logger.info(
        "Fold '{}': train={}, test={}, target='{}', correction='{}'",
        split_col, len(y_train), len(y_test), cfg.target_col, cfg.correction_method,
    )

    # Step 1: Batch correction (fit on train only)
    X_train, X_test = _apply_correction(X_train, X_test, batch_train, batch_test, cfg)

    # Step 2: PCA (fit on train only)
    X_train, X_test = _apply_pca(X_train, X_test, cfg)

    # Step 3: Train ML model
    train_fn = _MODEL_REGISTRY.get(model_name)
    if train_fn is None:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(_MODEL_REGISTRY.keys())}")

    probas_train, probas_test = train_fn(X_train, y_train, X_test, cfg)

    # Step 4: Evaluate
    thr, _, _ = youden_threshold(y_train, probas_train)
    metrics = evaluate_binary_classifier(y_test, probas_test, threshold=thr)

    result = metrics.to_dict()
    result["split"] = split_col
    result["model"] = model_name
    result["correction"] = cfg.correction_method
    result["target"] = cfg.target_col
    result["n_train"] = len(y_train)
    result["n_test"] = len(y_test)

    return result


def run_experiment(
    adata,
    cfg: FoldConfig,
    model_names: list[str] | None = None,
    n_splits: int = 10,
    split_prefix: str = "Split_seed_",
) -> pd.DataFrame:
    """Run full experiment: all folds × all models.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with raw embeddings, split columns, and targets.
    cfg : FoldConfig
        Pipeline configuration (shared across folds).
    model_names : list[str] or None
        Models to evaluate. Defaults to ``["tabpfn", "logreg", "lgbm"]``.
    n_splits : int
        Number of splits (seeds 0..n_splits-1).
    split_prefix : str
        Prefix for split columns.

    Returns
    -------
    pd.DataFrame
        One row per fold × model with all metrics.
    """
    if model_names is None:
        model_names = ["tabpfn", "logreg", "lgbm"]

    all_results = []

    for seed in range(n_splits):
        split_col = f"{split_prefix}{seed}"

        if split_col not in adata.obs.columns:
            logger.warning("Split column '{}' not found, skipping", split_col)
            continue

        # Run correction + PCA once per fold, cache features
        train_mask, test_mask = get_split_masks(adata, split_col)
        X_all = adata.obsm[cfg.embedding_key].astype(np.float32)
        y_all = adata.obs[cfg.target_col].values
        batch_all = adata.obs[BATCH_COL].values

        valid = pd.notna(y_all) & (y_all.astype(str) != "nan") & (y_all.astype(str) != "Missing")
        valid_vals = y_all[valid]
        if len(valid_vals) > 0:
            top_2_classes = pd.Series(valid_vals).value_counts().nlargest(2).index.values
            valid = valid & np.isin(y_all, top_2_classes)
            
        train_valid = train_mask & valid
        test_valid = test_mask & valid

        X_train_raw = X_all[train_valid]
        X_test_raw = X_all[test_valid]
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(y_all[valid].astype(str))
        y_train = le.transform(y_all[train_valid].astype(str)).astype(float)
        y_test = le.transform(y_all[test_valid].astype(str)).astype(float)
        batch_train = batch_all[train_valid]
        batch_test = batch_all[test_valid]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logger.warning(
                "Fold '{}': <2 classes in train or test. Skipping all models.",
                split_col,
            )
            continue

        logger.info(
            "\n{}\nFold '{}': train={}, test={}, target='{}', correction='{}'\n{}",
            "=" * 60, split_col, len(y_train), len(y_test),
            cfg.target_col, cfg.correction_method, "=" * 60,
        )

        # Shared transformation: correction + PCA (fit on train only)
        X_train_feat, X_test_feat = _apply_correction(
            X_train_raw, X_test_raw, batch_train, batch_test, cfg
        )
        X_train_feat, X_test_feat = _apply_pca(X_train_feat, X_test_feat, cfg)

        # Run each model on the same transformed features
        for model_name in model_names:
            logger.info("  Training model: {}", model_name)
            try:
                train_fn = _MODEL_REGISTRY[model_name]
                probas_train, probas_test = train_fn(
                    X_train_feat, y_train, X_test_feat, cfg
                )

                thr, _, _ = youden_threshold(y_train, probas_train)
                metrics = evaluate_binary_classifier(y_test, probas_test, threshold=thr)

                result = metrics.to_dict()
                result["split"] = split_col
                result["model"] = model_name
                result["correction"] = cfg.correction_method
                result["target"] = cfg.target_col
                result["n_train"] = len(y_train)
                result["n_test"] = len(y_test)
                all_results.append(result)

            except Exception as e:
                logger.error("  Model '{}' failed on '{}': {}", model_name, split_col, e)

    df = pd.DataFrame(all_results)
    return df
