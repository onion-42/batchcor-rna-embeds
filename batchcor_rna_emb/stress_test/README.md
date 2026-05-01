# Stress test & clinical benchmarks

This package evaluates **survival (C-index)** and **binary response (ROC-AUC)** on the training pool, plus **out-of-distribution** generalisation on public cohorts.

| Module | Purpose |
| --- | --- |
| **`survival_benchmark.py`** | **PFS / Harrell’s C-index only** — 5-fold CV. Writes **`metrics_csv/v4_survival_results.csv`** and executes **`metrics/metrics_tables.ipynb`**. |
| **`v4_definitive_pipeline.py`** | Full pipeline: survival + response + OOD + figures + leaderboard CSV under **`metrics_csv/`**; executes **`metrics/metrics_tables.ipynb`** at the end. |

Run survival-only (~15–20 min on CPU):

```bash
python -m batchcor_rna_emb.stress_test.survival_benchmark
```

Run everything:

```bash
python -m batchcor_rna_emb.stress_test.v4_definitive_pipeline
```

Re-run only the metrics notebook (after CSVs exist):

```bash
jupyter nbconvert --to notebook --execute --inplace metrics/metrics_tables.ipynb
```

Open **`metrics/metrics_tables.ipynb`** (**Run All**) for a short, styled summary; raw folds and F1 are in **`metrics_csv/`**.
