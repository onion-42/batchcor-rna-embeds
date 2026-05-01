"""
PFS survival benchmark — Harrell C-index only.

Loads TRAIN, builds fused features (embedding + clinical covariates), runs
5-fold stratified CV for each survival model. Does **not** run response
classification or OOD tests.

Usage::
    python -m batchcor_rna_emb.stress_test.survival_benchmark

Outputs::
    metrics_csv/v4_survival_results.csv
    metrics/metrics_tables.ipynb   (re-executed via nbconvert — open for tables)
"""

from __future__ import annotations

import scanpy as sc
import torch
from loguru import logger

from batchcor_rna_emb.stress_test.v4_definitive_pipeline import (
    CAE_KEY,
    METRICS_CSV_DIR,
    SCGPT_KEY,
    SEED,
    SURVIVAL_PCA_DIM,
    TRAIN_H5AD,
    build_features,
    build_survival_arrays,
    run_metrics_tables_notebook,
    save_survival_csv,
    set_seed,
    survival_cv,
)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("survival_benchmark — C-index only | device=%s", device)
    METRICS_CSV_DIR.mkdir(parents=True, exist_ok=True)

    train_ad = sc.read_h5ad(str(TRAIN_H5AD))
    mask, t_full, e_full = build_survival_arrays(train_ad)
    train_surv = train_ad[mask].copy()
    t_arr = t_full[mask].astype(float)
    e_arr = e_full[mask].astype(float)
    cohort_arr = train_surv.obs["Cohort"].astype(str).fillna("UNK").values

    feature_sets = [
        (CAE_KEY, SURVIVAL_PCA_DIM, f"cAE-PCA{SURVIVAL_PCA_DIM} + Clinical"),
        (SCGPT_KEY, SURVIVAL_PCA_DIM, f"Raw scGPT-PCA{SURVIVAL_PCA_DIM} + Clinical"),
        (CAE_KEY, None, "cAE-full + Clinical"),
    ]
    surv_summary: dict[str, dict] = {}
    for emb_key, pca_dim, label in feature_sets:
        X, feat_names, _ = build_features(
            train_surv, embedding_key=emb_key, pca_dim=pca_dim,
        )
        logger.info("[%s] feature matrix shape %s", label, X.shape)
        surv_summary[label] = survival_cv(
            X_full=X,
            t_full=t_arr,
            e_full=e_arr,
            cohort_full=cohort_arr,
            feat_names=feat_names,
            device=device,
            emb_label=label,
        )

    save_survival_csv(surv_summary)
    logger.success("Done — wrote %s", METRICS_CSV_DIR / "v4_survival_results.csv")
    run_metrics_tables_notebook()


if __name__ == "__main__":
    main()
