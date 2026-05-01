# BatchCorrect_RNA_embeds

End-to-end pipeline for evaluating how RNA-foundation-model embeddings
(scGPT) plus a Conditional Autoencoder batch correction (cAE) plus rich
clinical features predict clinical endpoints (Progression-Free Survival
and binary Response) on multi-cohort immunotherapy data.

**BG Internship 2026 — Group 7.**

---

## Headline result

| Task | Best feature set | Best model | Score |
|---|---|---|---|
| 5-fold CV C-index (PFS) | cAE-full + Clinical | DeepSurv MLP | **0.6616 ± 0.012** |
| 5-fold CV ROC-AUC (Response) | cAE-PCA32 + Clinical | Stacked ensemble | **0.6302 ± 0.023** |
| OOD ROC-AUC on PUB_ccRCC_ICI (n=526) | cAE + Clinical | LightGBM | **0.9612** |

The cAE-corrected embeddings beat raw scGPT on every survival model and
on two of the three out-of-distribution PUB cohorts. The model
generalises near-perfectly (AUC = 0.96) to a fully held-out ccRCC
immunotherapy cohort. Open **`metrics/metrics_tables.ipynb`** and **Run All**
for a compact, styled dashboard (the pipelines refresh it automatically).

---

## Setup

```powershell
python -m venv batcor_env
.\batcor_env\Scripts\Activate.ps1
pip install -e .
```

Required heavy dependencies: PyTorch, scanpy, lifelines, scikit-survival,
xgboost, lightgbm, loguru, matplotlib, seaborn, plotly. All listed in
`pyproject.toml`.

---

## Reproduce the entire pipeline

```powershell
# 1. Build per-cohort H5AD files from raw zarr + scGPT embeddings
python -m batchcor_rna_emb.modeling.pack_embeddings

# 2. Train the cAE (v3 balanced: latent=160, patience=20, lr=8e-4)
python -m batchcor_rna_emb.batch_correction.run_cae_correction

# 3. Generate the batch-correction visualisations
jupyter nbconvert --to notebook --execute --inplace `
    visualizations\visualizations_cae_correction.ipynb

# 4a. Survival / C-index only (~same runtime subset); writes survival CSV only
python -m batchcor_rna_emb.stress_test.survival_benchmark

# 4b. Run the full v4 evaluation pipeline (~20 min on CPU)
python -m batchcor_rna_emb.stress_test.v4_definitive_pipeline

# 5. (Optional) Re-execute metrics notebook only — pipelines already do this at exit
jupyter nbconvert --to notebook --execute --inplace metrics/metrics_tables.ipynb
```

Everything writes to **`metrics_csv/`** (numbers), **`metrics/`** (this notebook only),
and **`visualizations/`**. Random seed is
fixed at 42 across numpy, torch, scikit-learn, lifelines, sksurv,
xgboost and lightgbm — fully reproducible.

---

## Project layout

```
batchcor-rna-embeds/
├── README.md                       ← (this file)
├── pyproject.toml                  ← installable package metadata
│
├── batchcor_rna_emb/               ← main Python package
│   ├── batch_correction/
│   │   ├── cae.py                  ← Conditional AutoEncoder (frozen)
│   │   └── run_cae_correction.py   ← v3 balanced training
│   ├── modeling/
│   │   ├── pack_embeddings.py      ← raw zarr → H5AD
│   │   └── scgpt_embeddings.py     ← scGPT inference (frozen)
│   └── stress_test/
│       ├── README.md               ← survival vs full pipeline
│       ├── survival_benchmark.py   ← C-index–only benchmark
│       └── v4_definitive_pipeline.py    ← full evaluation entry-point
│
├── data/processed/                 ← H5AD inputs (TRAIN + 3 PUB cohorts)
├── checkpoints/cae_trained.pt      ← trained cAE weights (v3)
├── pretrained/                     ← scGPT pretrained weights
│
├── metrics_csv/                    ← all CSV + batch_correction_metrics.json
│   ├── v4_survival_results.csv
│   ├── v4_classification_results.csv
│   ├── v4_ood_pub_results.csv
│   ├── v4_final_leaderboard.csv
│   └── batch_correction_metrics.{csv,json}
│
├── metrics/                        ← **only** metrics_tables.ipynb (tables viewer)
│   └── metrics_tables.ipynb
│
└── visualizations/                 ← PNG figures (no interactive HTML in repo)
    ├── visualizations_cae_correction.ipynb   ← cAE quality notebook
    ├── batch_correction_metrics.png
    ├── per_cohort_silhouette.png
    ├── TRAIN_*.png
    ├── v4_cindex_bar.png
    ├── v4_response_auc_bar.png
    ├── v4_pub_ood_auc.png
    ├── v4_km_risk_strata.png
    └── v4_feature_importance.png
```

---

## Methodology in one paragraph

The scGPT transformer encodes each patient's bulk-RNA profile into a
512-D embedding. A Conditional AutoEncoder (cAE, v3 balanced) is trained
on those embeddings with a one-hot Cohort code as a conditioning signal
and projects them into a domain-aligned 512-D space (`cAE_embedding`).
For OOD evaluation, public cohorts that the cAE has never seen are
projected with neutral one-hots in both encoder and decoder
(`cAE_embedding_OOD`). The downstream evaluation fuses the corrected
embedding with 47 dense numeric clinical features (4 MFP scores + 43
Kassandra cell-type fractions), one-hot categorical features
(Diagnosis, Cohort, Therapy_group, MSKCC risk, Stage, Gender) and
sparse numerics with explicit missing-indicators (Age, TMB, PDL1).
Six survival models (Cox-strat, Coxnet, RSF, GB-Surv, XGBoost-Cox,
DeepSurv) and six classifiers (LogReg, RF, XGBoost, LightGBM, MLP,
stack) are run with 5-fold stratified CV and a global ensemble is
trained on the entire pool to evaluate out-of-distribution AUC on the
three public PUB cohorts.
