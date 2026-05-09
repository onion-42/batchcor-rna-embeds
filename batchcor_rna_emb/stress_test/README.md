# Stress test & clinical benchmarks

This package evaluates **survival (C-index)** and **binary response (ROC-AUC)** on the training pool, plus **out-of-distribution** generalisation on public cohorts.

| Module | Purpose |
| --- | --- |
| **`survival_benchmark.py`** | **PFS / Harrell’s C-index only** — 5-fold CV. Writes **`metrics_csv/v4_survival_results.csv`**, regenerates **`v4_cindex_survival_matrix.csv`**, and executes **`metrics/metrics_tables.ipynb`**. |
| **`v4_definitive_pipeline.py`** | Full pipeline: survival + response + OOD + figures + leaderboard CSV under **`metrics_csv/`**; executes **`metrics/metrics_tables.ipynb`** at the end. |

Reproducibility: set environment variable **`V4_SEED`** (integer) before launch to
match multi-seed experiments; default is **42**. Set **`V4_SMOKE=1`** to collapse
to 2 folds + 10-epoch DeepSurv/MLP for quick local checks (~5–6 min vs ~20 min).

CV honesty: every fold of `survival_cv` and `classification_cv` re-fits the PCA
basis, scaler, clinical-feature medians, and one-hot category dictionary on
training rows only — no embedding/scaler/imputer leakage from validation rows.

Auxiliary CSVs written by `v4_definitive_pipeline` next to the main results:

| File | Shape |
| --- | --- |
| `v4_response_auc_matrix.csv` | wide: feature set × classifier (mean ROC-AUC) |
| `v4_per_pub_best.csv` | one row per PUB cohort: best (embedding, classifier) by AUC |
| `v4_km_risk_scores.csv` | per-patient risk score / tertile feeding `v4_km_risk_strata.png` |

Run survival-only (~15–20 min on CPU):

```bash
python -m batchcor_rna_emb.stress_test.survival_benchmark
```

Run everything:

```bash
python -m batchcor_rna_emb.stress_test.v4_definitive_pipeline
# Quick local smoke (~5–6 min, weak metrics — diagnostic only):
V4_SMOKE=1 python -m batchcor_rna_emb.stress_test.v4_definitive_pipeline
```

Re-run only the metrics notebook (after CSVs exist):

```bash
jupyter nbconvert --to notebook --execute --inplace metrics/metrics_tables.ipynb
```

Open **`metrics/metrics_tables.ipynb`** (**Run All**) for a short, styled summary; raw folds and F1 are in **`metrics_csv/`**.
