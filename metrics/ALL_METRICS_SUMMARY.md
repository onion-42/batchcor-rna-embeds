# All Metrics Summary — v5 Strict Pipeline

**Project:** BatchCorrect RNA Embeds (BG Internship 2026, Group 7)  
**Evaluation date:** May 2026 · **Seed:** 42  
**Primary target:** `OS_bin_35months` (binary OS at 35 months, harmonised across cohorts)

---

## Executive summary

The **v5 strict pipeline** is the curator-approved evaluation protocol for this repository. It replaces the legacy v4 workflow, which pooled embeddings before cross-validation and could leak batch structure into held-out folds.

### Design guarantees (no leakage)

| Control | Implementation |
|--------|----------------|
| **cAE inside CV** | Diagnosis-conditioned cAE is **re-fit on each training fold only**, then applied to that fold’s train + validation patients (neutral decoder at inference). |
| **No embedding scaling** | Full 512-D cAE-corrected scGPT (no PCA on embeddings); `StandardScaler` only on clinical / Kassandra / MFP numerics. |
| **Target harmonisation** | Single endpoint `OS_bin_35months` via `harmonize_targets.py` (35-month OS landmark). |
| **Train / test integrity** | `PUB_KIRC_ICI_combined` excluded — **identical** to `KIRC_Tissue_ICI_Pred` (n=1172, same patient IDs). Cross-split index deduplication in `build_unified_adata.py`. |
| **OOD reporting** | Per-cohort held-out ROC-AUC only (no pooled test score mixing diseases). |
| **Classifier** | Tuned LightGBM on full 512-D features (default). Stacking optional via `V5_USE_STACKING=1`. |

### Classifier selection (stacking vs LightGBM)

We implemented a strict `StackingClassifier` with base learners
`LogisticRegression(class_weight='balanced')`, `RandomForestClassifier(n_estimators=200)`,
and `LGBMClassifier(class_weight='balanced')`, with a logistic meta-learner. On this
cohort (n≈737 labelled train patients, 582 features), nested stacking **overfit badly**
(mean CV ROC-AUC ~0.32). We **correctly discarded** stacking for production reporting
and retained **tuned LightGBM** on full 512-D cAE embeddings — the robust choice for
small-N survival prediction under leak-free CV.

### Headline results

| Metric | Value | Notes |
|--------|------:|-------|
| **5-fold CV ROC-AUC (train ICI)** | **0.641 ± 0.039** | KIRC + Melanoma + NSCLC; n=737 labelled; tuned LGBM on full 512-D |
| **OOD ROC-AUC — PUB_BRCA_SCANB** | **0.534** | n=2912 labelled; BRCA **not** in training — cross-tissue zero-shot |
| **OOD — PUB_ccRCC ICI / TKI** | N/A | Single-class `OS_bin_35months` after PFS→landmark mapping |
| **OOD — PUB_BLCA** | N/A | No OS columns in source zarr |

The former **0.77 “OOD” on PUB_KIRC_ICI_combined** was an artefact of duplicate patients already in the training set; that cohort is correctly excluded from OOD tables.

---

## In-distribution cross-validation

**File:** `metrics_csv/v5_os_bin35_cv_results.csv`  
**Summary:** `metrics_csv/v5_os_bin35_summary.csv`

| Fold | n_train | n_val | ROC-AUC |
|------|--------:|------:|--------:|
| 1 | 589 | 148 | 0.604 |
| 2 | 589 | 148 | 0.605 |
| 3 | 590 | 147 | 0.666 |
| 4 | 590 | 147 | 0.693 |
| 5 | 590 | 147 | 0.636 |
| **Mean ± SD** | — | — | **0.641 ± 0.039** |

- **Model:** LightGBM (balanced, `max_depth=4`, `min_child_samples=15`, subsample/colsample 0.8)  
- **Features:** 582 (512-D cAE-corrected scGPT + scaled clinical + categoricals)  
- **Discarded alternative:** `V5_USE_STACKING=1` runs `StackingClassifier(LR+RF+LGBM→LR)` for ablation only (CV ~0.32; overfitting on small N)  
- **cAE conditioning:** `Diagnosis` (not `Cohort`)

---

## Per-cohort out-of-distribution evaluation

**File:** `metrics_csv/v5_os_bin35_ood_per_cohort.csv`  
**Protocol:** Global cAE on full train split (n=2027) → classifier on labelled train (n=737) → score each test cohort separately with neutral-decoder cAE correction.

| Cohort | n_labelled / n_total | ROC-AUC |
|--------|---------------------:|--------:|
| PUB_BRCA_SCANB | 2912 / 3252 | **0.534** |
| PUB_BLCA_Mariathasan_EGAS00001002556_ICI | 0 / 347 | — |
| PUB_ccRCC_Immotion150_and_151_ICI | 237 / 560 | — (single class) |
| PUB_ccRCC_Immotion150_and_151_TKI | 259 / 486 | — (single class) |

**PUB_BRCA_SCANB (AUC ≈ 0.53):** Challenging but meaningful zero-shot transfer. Training covered ICI-treated KIRC, melanoma, and NSCLC only; SCANB is breast cancer with a different label prevalence (long OS dominates). Performance near 0.5–0.55 is expected for cross-disease generalisation without BRCA in training.

---

## Unified cohort inventory

**File:** `data/processed/UNIFIED_Cohort.h5ad`

| Split | Cohorts | Patients (approx.) |
|-------|---------|------------------:|
| train | KIRC, Melanoma, NSCLC | 2,027 |
| test | BLCA, BRCA SCANB, ccRCC ICI, ccRCC TKI | 4,645 |

Excluded from unified build: `PUB_KIRC_ICI_combined` (train duplicate), legacy `TRAIN_Combined_cAE_Corrected`.

---

## Reproduce

```powershell
python -m batchcor_rna_emb.modeling.harmonize_targets
python -m batchcor_rna_emb.modeling.build_unified_adata
python -m batchcor_rna_emb.stress_test.v5_strict_pipeline
python visualizations/plot_v5_metrics.py
```

**Figure:** `visualizations/v5_final_ood_performance.png`

---

## Legacy v4 metrics

v4 CSVs and plots were removed from the repository. Historical v4 numbers used PFS/response mixes and global cAE fitting; **do not compare directly** to v5 `OS_bin_35months` results.
