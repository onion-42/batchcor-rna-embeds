"""
One-shot helper: regenerate `v4_cindex_bar.png` and `v4_response_auc_bar.png`
from the latest CSVs without re-running the full 23-min v4 pipeline.

Use after editing the bar-plot styling in `v4_definitive_pipeline.py`.

    python scripts/refresh_v4_bar_plots.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from batchcor_rna_emb.stress_test.v4_definitive_pipeline import (
    METRICS_CSV_DIR,
    SURVIVAL_MODEL_COL_ORDER,
    V4_CLF_CSV,
    V4_SURV_CSV,
    viz_cindex_bar,
    viz_response_auc_bar,
)


def _surv_long_to_summary(df: pd.DataFrame) -> dict:
    fold_cols = [c for c in df.columns if c.startswith("fold_")]
    out: dict = {}
    for emb, sub in df.groupby("embedding"):
        out[emb] = {}
        for _, row in sub.iterrows():
            folds = [row[c] for c in fold_cols]
            out[emb][row["model"]] = {
                "folds": folds,
                "mean" : float(row["cindex_mean"]),
                "std"  : float(row["cindex_std"]),
            }
    # Preserve discovery order (cAE-PCA32, Raw-PCA32, cAE-full)
    return {k: out[k] for k in df["embedding"].drop_duplicates()}


def _clf_long_to_summary(df: pd.DataFrame) -> dict:
    auc_cols = [c for c in df.columns if c.startswith("auc_fold_")]
    f1_cols  = [c for c in df.columns if c.startswith("f1_fold_")]
    out: dict = {}
    for emb, sub in df.groupby("embedding"):
        out[emb] = {}
        for _, row in sub.iterrows():
            out[emb][row["model"]] = {
                "auc_folds": [row[c] for c in auc_cols],
                "f1_folds" : [row[c] for c in f1_cols],
                "auc_mean" : float(row["auc_mean"]),
                "auc_std"  : float(row["auc_std"]),
                "f1_mean"  : float(row["f1_mean"]),
                "f1_std"   : float(row["f1_std"]),
            }
    return {k: out[k] for k in df["embedding"].drop_duplicates()}


def main() -> None:
    surv_df = pd.read_csv(V4_SURV_CSV)
    clf_df  = pd.read_csv(V4_CLF_CSV)

    surv_summary = _surv_long_to_summary(surv_df)
    clf_summary  = _clf_long_to_summary(clf_df)

    # Sanity-check we have all expected models in the survival summary
    for emb, models in surv_summary.items():
        missing = [m for m in SURVIVAL_MODEL_COL_ORDER if m not in models]
        if missing:
            print(f"  warn: {emb} missing models {missing}")

    viz_cindex_bar(surv_summary)
    viz_response_auc_bar(clf_summary)
    print("OK - regenerated v4_cindex_bar.png + v4_response_auc_bar.png")


if __name__ == "__main__":
    main()
