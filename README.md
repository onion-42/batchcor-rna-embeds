# BatchCorrect_RNA_embeds

End-to-end pipeline for evaluating how RNA foundation-model embeddings (scGPT),
diagnosis-conditioned conditional autoencoder (cAE) batch correction, and clinical
features predict **35-month overall survival** (`OS_bin_35months`) on multi-cohort
immunotherapy and public oncology datasets.

**BG Internship 2026 — Group 7.**

---

## Executive summary (v5 strict pipeline)

The **v5 strict pipeline** is the authoritative evaluation for this repository. It
addresses curator feedback on leakage, target harmonisation, and honest
out-of-distribution (OOD) reporting.

### What makes v5 rigorous

1. **cAE inside 5-fold CV** — The diagnosis-conditioned cAE is trained **only on each
   CV training fold**, then used to correct that fold’s train and validation embeddings.
   There is no global cAE fit across all training data before cross-validation.

2. **Target harmonisation** — All cohorts expose a single binary endpoint
   `OS_bin_35months` (35-month OS landmark), computed by `harmonize_targets.py` from
   harmonised OS/PFS time and event columns.

3. **No train/test duplicates** — `PUB_KIRC_ICI_combined` is **excluded** from unified
   data and OOD metrics because it is **identical** to `KIRC_Tissue_ICI_Pred` (n=1172,
   same patient IDs). `build_unified_adata.py` also drops any test row whose index
   already appears in train.

4. **Per-cohort OOD only** — After CV, a final model is trained on labelled train
   patients; each **test cohort is scored separately** (no pooled BRCA + kidney AUC).

5. **Feature hygiene** — PCA on embeddings without `StandardScaler`; clinical/Kassandra
   numerics scaled only; survival/response columns blocklisted from `X`.

### Headline results

| Metric | Value |
|--------|------:|
| **5-fold CV ROC-AUC** (train ICI, `OS_bin_35months`) | **0.639 ± 0.024** |
| **OOD ROC-AUC — PUB_BRCA_SCANB** (zero-shot, BRCA not in train) | **0.544** |

**PUB_BRCA_SCANB (~0.54):** Expected for **cross-tissue zero-shot** transfer. Training
cohorts are ICI-treated KIRC, melanoma, and NSCLC only; SCANB is breast cancer with a
different survival landscape. Modest AUC above chance reflects partial signal in shared
clinical and embedding structure, not in-distribution performance.

**Note on PUB_KIRC_ICI_combined:** An earlier pooled evaluation reported ~0.77 OOD AUC
on this cohort; that was **inflated by duplicate patients** already in the training set.
v5 correctly excludes this cohort from OOD tables.

**Artifacts:** `metrics_csv/v5_os_bin35_*.csv` · `visualizations/v5_final_ood_performance.png`

---

## Setup

```powershell
python -m venv batcor_env
.\batcor_env\Scripts\Activate.ps1
pip install -e .
```

Dependencies: PyTorch, scanpy, anndata, lightgbm, scikit-learn, loguru, matplotlib.
See `pyproject.toml`.

---

## Reproduce the v5 pipeline

```powershell
# 1. Harmonise OS_bin_35months on all raw zarr cohorts
python -m batchcor_rna_emb.modeling.harmonize_targets

# 2. Per-cohort h5ad (if not already built)
python -m batchcor_rna_emb.modeling.pack_embeddings

# 3. Unified AnnData (train + test, deduplicated)
python -m batchcor_rna_emb.modeling.build_unified_adata

# 4. Strict CV + per-cohort OOD (~2 min CPU)
python -m batchcor_rna_emb.stress_test.v5_strict_pipeline

# 5. Summary figure
python visualizations/plot_v5_metrics.py

# Optional: slim export zip for merging embeddings with other groups
python scripts/export_embeddings_for_merge.py
```

Environment flags:

| Variable | Effect |
|----------|--------|
| `V5_SEED` | RNG seed (default 42) |
| `V5_SMOKE=1` | 2-fold CV, fewer cAE epochs (quick dev) |
| `V5_UNIFIED` | Path to unified h5ad override |

Full metric narrative: [`metrics/ALL_METRICS_SUMMARY.md`](metrics/ALL_METRICS_SUMMARY.md).

---

## Project layout

```
batchcor-rna-embeds/
├── README.md
├── metrics/
│   └── ALL_METRICS_SUMMARY.md          ← v5 results narrative
├── batchcor_rna_emb/
│   ├── modeling/
│   │   ├── cohort_registry.py          ← train / test / duplicate exclusions
│   │   ├── harmonize_targets.py        ← OS_bin_35months
│   │   ├── build_unified_adata.py      ← UNIFIED_Cohort.h5ad
│   │   ├── pack_embeddings.py
│   │   └── scgpt_embeddings.py
│   ├── batch_correction/
│   │   └── cae.py                      ← diagnosis-conditioned cAE
│   └── stress_test/
│       └── v5_strict_pipeline.py       ← primary evaluation entry-point
├── data/
│   ├── raw/                            ← *.adata.zarr per cohort
│   └── processed/                      ← per-cohort + UNIFIED_Cohort.h5ad
├── metrics_csv/
│   ├── v5_os_bin35_cv_results.csv
│   ├── v5_os_bin35_summary.csv
│   └── v5_os_bin35_ood_per_cohort.csv
├── visualizations/
│   ├── plot_v5_metrics.py
│   └── v5_final_ood_performance.png
└── scripts/
    └── export_embeddings_for_merge.py
```

Legacy **v4** metrics, plots, and `v4_definitive_pipeline.py` are deprecated; v4 CSVs
were removed from the repo.

---

## Train vs test cohorts

| Split | Cohorts |
|-------|---------|
| **train** | `KIRC_Tissue_ICI_Pred`, `Melanoma_Tissue_ICI_Pred`, `NSCLC_Tissue_ICI_Pred` |
| **test** | `PUB_BRCA_SCANB`, `PUB_BLCA_*`, `PUB_ccRCC_*` (ICI + TKI) |
| **excluded** | `PUB_KIRC_ICI_combined` (duplicate of train KIRC) |

---

## Methodology (v5)

scGPT produces 512-D embeddings per patient. Within each CV fold, a cAE conditioned on
**Diagnosis** removes batch-specific variation; at inference the decoder uses **neutral
(zero) batch vectors** so corrected embeddings represent shared biology. LightGBM
predicts `OS_bin_35months` from PCA-reduced embeddings plus scaled clinical features.
OOD evaluation reuses one global cAE and classifier fit on all labelled train data,
then applies the same protocol independently to each external cohort.

---

## Sharing embeddings

`data/exports/embeddings_to_merge.zip` contains slim h5ad files (patient metadata +
`obsm['scGPT_embedding']`) for collaborator merge. See `data/exports/embeddings_to_merge/SHARE.md`.
