"""
run_cae_correction.py
=====================
Execution script for the Conditional Autoencoder (cAE) batch-correction
pipeline.  Operates on the three TRAINING clinical cohorts, corrects the
scGPT embeddings across domains, and produces dimensionality-reduced
projections for downstream analysis and visualisation.

Pipeline stages
---------------
  1. Load & annotate  – read per-cohort .h5ad files, tag with cohort label
  2. Concatenate       – merge into a single joint AnnData
  3. Batch labels      – encode cohort strings → contiguous integer indices
  4. Train cAE         – learn domain-invariant latent space
  5. Save model        – checkpoint the trained cAE weights to disk
  6. Correct           – encode(true_OH) → decode(zero_OH)
  7. Projections       – 3-D UMAP + 128-D PCA on corrected embeddings
  8. Save AnnData      – write fully-annotated joint object to disk

Improvements over v1
---------------------
  - Typed return signature on run_cae_training (ConditionalAutoencoder).
  - RNG seeded before UMAP and PCA for reproducible projections.
  - UMAP n_neighbors adapted to cohort size: min(15, n-1) avoids crash.
  - obs_names_make_unique() removed after ad.concat because obs_names were
    already prefixed per-cohort in load_and_annotate_cohorts; calling it
    again would silently append "-1" suffixes to legitimate patient IDs.
  - Trained model checkpoint saved to disk (Stage 5) so a downstream crash
    (e.g. OOM during UMAP) does not force retraining.

Usage
-----
    python run_cae_correction.py

    # Override hyper-parameters at runtime:
    CAE_LATENT_DIM=256 CAE_MAX_EPOCHS=300 CAE_SEED=0 python run_cae_correction.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import torch
import umap as umap_lib
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from batchcor_rna_emb.batch_correction.cae import (
    CAEConfig,
    ConditionalAutoencoder,
    correct_embeddings,
    train_cae,
)

# =============================================================================
# LOGGING
# =============================================================================
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | <cyan>{function}</cyan> | {message}"
    ),
    level="INFO",
)

# =============================================================================
# PATHS
# =============================================================================
PROCESSED_DIR  = Path("data/processed")
OUTPUT_PATH    = PROCESSED_DIR / "TRAIN_Combined_cAE_Corrected.h5ad"
MODEL_CKPT_DIR = Path("checkpoints")    # cAE weights saved here after training

# Each tuple: (h5ad filename, short cohort label stored in .obs['cohort'])
TRAINING_COHORTS: list[tuple[str, str]] = [
    ("KIRC_Tissue_ICI_Pred.h5ad",    "KIRC"),
    ("Melanoma_Tissue_ICI_Pred.h5ad", "Melanoma"),
    ("NSCLC_Tissue_ICI_Pred.h5ad",   "NSCLC"),
]

# .obsm keys
SCGPT_OBSM_KEY : str = "scGPT_embedding"
CAE_OBSM_KEY   : str = "cAE_embedding"
UMAP_OBSM_KEY  : str = "UMAP3d_cAE_embedding"
PCA_OBSM_KEY   : str = "PCA128d_cAE_embedding"

# Projection hyper-parameters
UMAP_N_COMPONENTS : int = 3
PCA_N_COMPONENTS  : int = 128

# =============================================================================
# CAE HYPER-PARAMETERS
# =============================================================================
# NOTE: latent_dim / patience / weight_decay / lr are HARDCODED to the
# "balanced (Goldilocks)" preset chosen after sweeping two extremes:
#   * v1 (hard)  : latent=128, patience=30, wd=1e-4, lr=1e-3
#                  -> over-corrected. ARI 0.85 -> 0.69, NMI 0.69 -> 0.52,
#                  downstream AUC fell.
#   * v2 (soft)  : latent=384, patience=10, wd=1e-3, lr=5e-4
#                  -> too soft (near-identity). Biology preserved (ARI 0.97)
#                  but cohorts not actually mixed (silhouette -0.16 -> +0.85).
#
# v3 (balanced) sits between the two:
#   - LATENT_DIM=160    : enough compression to act as a real bottleneck,
#                         but generous enough to keep cross-tissue biology.
#   - PATIENCE=20       : medium training horizon — long enough to find a
#                         shared latent space, short enough to avoid collapse.
#   - WEIGHT_DECAY=5e-4 : moderate L2 — decoder is constrained but not frozen.
#   - LR=8e-4           : middle-of-the-road learning rate.
# Only MAX_EPOCHS / BATCH_SIZE / SEED remain env-var overridable.
_LATENT_DIM   = 160
_PATIENCE     = 20
_WEIGHT_DECAY = 5e-4
_LR           = 8e-4

_MAX_EPOCHS   = int(os.environ.get("CAE_MAX_EPOCHS",    500))
_BATCH_SIZE   = int(os.environ.get("CAE_BATCH_SIZE",    128))
_SEED         = int(os.environ.get("CAE_SEED",            42))


# =============================================================================
# STAGE 1 — LOAD & ANNOTATE
# =============================================================================

def load_and_annotate_cohorts(
    cohorts  : list[tuple[str, str]],
    data_dir : Path,
    obsm_key : str,
) -> list[ad.AnnData]:
    """
    Load each cohort .h5ad file, attach the cohort string label, and
    prefix obs_names with the cohort label to guarantee global uniqueness.

    Validates that the scGPT embedding key exists before returning so
    failures surface immediately with a clear message.
    """
    objects: list[ad.AnnData] = []

    for filename, label in cohorts:
        path = data_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Training cohort file not found: {path}\n"
                "Ensure PROCESSED_DIR is correct and the scGPT embedding "
                "script has already been run."
            )

        logger.info(f"Loading cohort '{label}' from: {path}")
        adata = sc.read_h5ad(str(path))
        logger.info(
            f"  → {adata.n_obs:,} patients × {adata.n_vars:,} features"
        )

        if obsm_key not in adata.obsm:
            raise KeyError(
                f"Expected .obsm['{obsm_key}'] in {filename} but found: "
                f"{list(adata.obsm.keys())}\n"
                "Run the scGPT embedding script first."
            )

        emb = adata.obsm[obsm_key]
        logger.info(
            f"  → scGPT embedding: shape={emb.shape}  dtype={emb.dtype}"
        )

        adata.obs["cohort"] = label
        # Prefix obs_names here — do NOT call obs_names_make_unique() later
        # because that appends "-1", "-2" … suffixes that corrupt patient IDs.
        adata.obs_names = [f"{label}_{n}" for n in adata.obs_names]

        objects.append(adata)

    return objects


# =============================================================================
# STAGE 2 — CONCATENATE
# =============================================================================

def concatenate_cohorts(objects: list[ad.AnnData]) -> ad.AnnData:
    """
    Merge per-cohort AnnData objects into one joint object.

    merge='same'  – keeps only .var columns that are identical across all
                    cohorts, preventing NaN-filled annotation tables.
    join='outer'  – preserves cohort-specific .obs columns (filled with NaN
                    for other cohorts); safe because downstream analysis uses
                    .obsm, not .X.

    obs_names_make_unique() is intentionally NOT called: obs_names were
    already globally unique after per-cohort prefixing in Stage 1.
    """
    total = sum(o.n_obs for o in objects)
    logger.info(
        f"Concatenating {len(objects)} cohorts ({total:,} patients total) …"
    )

    joint = ad.concat(
        objects,
        axis     = 0,
        join     = "outer",
        merge    = "same",
        uns_merge= "same",
        label    = None,   # cohort already in .obs['cohort']
    )

    logger.info(
        f"Joint AnnData: {joint.n_obs:,} patients × {joint.n_vars:,} features"
    )
    logger.info("Cohort composition:")
    for cohort, count in joint.obs["cohort"].value_counts().sort_index().items():
        logger.info(f"  {cohort:<20} {count:>5,} patients")

    return joint


# =============================================================================
# STAGE 3 — BATCH LABELS
# =============================================================================

def encode_batch_labels(adata: ad.AnnData) -> np.ndarray:
    """
    Encode .obs['cohort'] strings → contiguous int64 indices in .obs['batch_idx'].

    Encoding is alphabetical for deterministic reproducibility across runs.
    The mapping is logged so you can verify the one-hot layout.

    Returns
    -------
    batch_idx : int64 numpy array of shape (n_obs,)
    """
    le = LabelEncoder()
    le.fit(sorted(adata.obs["cohort"].unique()))
    batch_idx = le.transform(adata.obs["cohort"]).astype(np.int64)
    adata.obs["batch_idx"] = batch_idx

    logger.info("Cohort → integer batch label mapping:")
    for i, cls in enumerate(le.classes_):
        n = int((batch_idx == i).sum())
        logger.info(f"  [{i}]  {cls:<20}  {n:>5,} patients")

    return batch_idx


# =============================================================================
# STAGE 4 — TRAIN cAE
# =============================================================================

def run_cae_training(
    raw_embs  : np.ndarray,
    batch_idx : np.ndarray,
    n_batches : int,
    emb_dim   : int,
) -> ConditionalAutoencoder:       # FIX: typed return instead of `object`
    """
    Build a CAEConfig scaled to emb_dim and train the cAE.

    hidden_dims are proportionally scaled so the architecture is appropriate
    regardless of whether scGPT_human outputs 512 or 768 dimensions.
    """
    h1 = max(256, int(emb_dim * 0.75))   # 384 @ 512-dim,  576 @ 768-dim
    h2 = max(128, int(emb_dim * 0.50))   # 256 @ 512-dim,  384 @ 768-dim

    cfg = CAEConfig(
        emb_dim      = emb_dim,
        n_batches    = n_batches,
        latent_dim   = _LATENT_DIM,
        hidden_dims  = [h1, h2],
        lr           = _LR,
        weight_decay = _WEIGHT_DECAY,
        batch_size   = _BATCH_SIZE,
        max_epochs   = _MAX_EPOCHS,
        patience     = _PATIENCE,
        seed         = _SEED,
        normalize_emb= True,
    )

    logger.info(
        f"CAEConfig | emb_dim={emb_dim} | n_batches={n_batches} | "
        f"latent_dim={cfg.latent_dim} | hidden_dims={cfg.hidden_dims} | "
        f"lr={cfg.lr} | max_epochs={cfg.max_epochs} | "
        f"patience={cfg.patience} | seed={cfg.seed}"
    )

    t0    = time.perf_counter()
    model = train_cae(raw_embs, batch_idx, cfg=cfg)
    logger.info(f"cAE training complete in {time.perf_counter() - t0:.1f}s.")
    return model


# =============================================================================
# STAGE 5 — SAVE MODEL CHECKPOINT
# =============================================================================

def save_model_checkpoint(
    model     : ConditionalAutoencoder,
    ckpt_dir  : Path,
    filename  : str = "cae_trained.pt",
) -> Path:
    """
    Persist the trained cAE weights so a downstream crash (e.g. OOM during
    UMAP) does not force a full retraining run.

    Saved bundle contains both state_dict and the config so the model can
    be reconstructed from scratch without keeping the Python object alive.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / filename

    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg"        : model.cfg,
        },
        str(ckpt_path),
    )
    size_kb = ckpt_path.stat().st_size / 1024
    logger.info(f"Model checkpoint saved → {ckpt_path}  ({size_kb:.0f} KB)")
    return ckpt_path


# =============================================================================
# STAGE 6 — BATCH CORRECTION (INFERENCE)
# =============================================================================

def run_correction(
    model     : ConditionalAutoencoder,
    raw_embs  : np.ndarray,
    batch_idx : np.ndarray,
) -> np.ndarray:
    """
    Apply the trained cAE: encode with true one-hot, decode with zero vector.
    """
    logger.info(f"Running batch correction on {len(raw_embs):,} patients …")
    t0 = time.perf_counter()
    corrected = correct_embeddings(
        model         = model,
        embeddings    = raw_embs,
        batch_indices = batch_idx,
        batch_size    = _BATCH_SIZE * 4,   # inference can use 4× training size
    )
    logger.info(
        f"Correction complete in {time.perf_counter() - t0:.1f}s | "
        f"shape={corrected.shape} | dtype={corrected.dtype}"
    )
    return corrected


# =============================================================================
# STAGE 7 — DIMENSIONALITY REDUCTION
# =============================================================================

def run_umap_3d(emb: np.ndarray, seed: int = _SEED) -> np.ndarray:
    """
    Fit a 3-D UMAP on the cAE-corrected embeddings.

    n_neighbors is adapted to cohort size: min(15, n_samples - 1) prevents
    a crash when a tiny cohort has fewer than 16 patients.
    """
    n = len(emb)
    n_neighbors = min(15, n - 1)   # FIX: adaptive n_neighbors

    logger.info(
        f"Computing {UMAP_N_COMPONENTS}-D UMAP | "
        f"n={n:,} | n_neighbors={n_neighbors} | seed={seed}"
    )
    t0 = time.perf_counter()

    # FIX: seed numpy before UMAP for reproducible layout
    np.random.seed(seed)

    reducer = umap_lib.UMAP(
        n_components = UMAP_N_COMPONENTS,
        n_neighbors  = n_neighbors,
        min_dist     = 0.3,
        metric       = "euclidean",
        random_state = seed,
        verbose      = False,
    )
    coords = reducer.fit_transform(emb).astype(np.float32)
    logger.info(
        f"UMAP complete in {time.perf_counter() - t0:.1f}s | "
        f"shape={coords.shape}"
    )
    return coords


def run_pca(
    emb         : np.ndarray,
    n_components: int = PCA_N_COMPONENTS,
    seed        : int = _SEED,
) -> np.ndarray:
    """
    Fit a truncated PCA on the cAE-corrected embeddings.

    n_components is capped at min(n_samples, emb_dim, n_components) so the
    call does not crash on tiny cohorts.
    """
    effective_n = min(n_components, emb.shape[0], emb.shape[1])
    if effective_n < n_components:
        logger.warning(
            f"PCA n_components reduced from {n_components} to {effective_n} "
            f"(data shape {emb.shape[0]} × {emb.shape[1]})."
        )

    logger.info(
        f"Computing {effective_n}-D PCA | n={len(emb):,} | seed={seed}"
    )
    t0 = time.perf_counter()

    # FIX: seed numpy before PCA for reproducible SVD initialisation
    np.random.seed(seed)

    pca    = PCA(n_components=effective_n, random_state=seed)
    coords = pca.fit_transform(emb).astype(np.float32)
    var_ex = pca.explained_variance_ratio_.sum()
    logger.info(
        f"PCA complete in {time.perf_counter() - t0:.1f}s | "
        f"shape={coords.shape} | cumulative_var={100 * var_ex:.1f}%"
    )
    return coords


# =============================================================================
# STAGE 8 — SAVE AnnData
# =============================================================================

def save_result(adata: ad.AnnData, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving corrected joint AnnData → {output_path}")
    t0 = time.perf_counter()
    adata.write_h5ad(str(output_path), compression="gzip")
    size_mb = output_path.stat().st_size / 1024 ** 2
    logger.info(
        f"Saved in {time.perf_counter() - t0:.1f}s | size={size_mb:.1f} MB"
    )


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main() -> None:
    wall_start = time.perf_counter()

    logger.info("=" * 65)
    logger.info("cAE Batch Correction Pipeline — TRAIN cohorts")
    logger.info(f"Global seed: {_SEED}")
    logger.info("=" * 65)

    # Stage 1 — Load & annotate
    logger.info("[Stage 1/8] Loading and annotating cohort files …")
    cohort_objects = load_and_annotate_cohorts(
        cohorts  = TRAINING_COHORTS,
        data_dir = PROCESSED_DIR,
        obsm_key = SCGPT_OBSM_KEY,
    )

    # Stage 2 — Concatenate
    logger.info("[Stage 2/8] Concatenating cohorts …")
    joint = concatenate_cohorts(cohort_objects)
    del cohort_objects   # release per-cohort copies

    # Stage 3 — Batch labels
    logger.info("[Stage 3/8] Encoding cohort labels → integer batch_idx …")
    batch_idx = encode_batch_labels(joint)
    n_batches = int(batch_idx.max()) + 1

    raw_embs = np.asarray(joint.obsm[SCGPT_OBSM_KEY], dtype=np.float32)
    emb_dim  = raw_embs.shape[1]
    logger.info(
        f"Raw embeddings: shape={raw_embs.shape} | "
        f"dtype={raw_embs.dtype} | n_batches={n_batches}"
    )

    # Stage 4 — Train cAE
    logger.info("[Stage 4/8] Training Conditional Autoencoder …")
    model = run_cae_training(raw_embs, batch_idx, n_batches, emb_dim)

    # Stage 5 — Checkpoint the trained model
    logger.info("[Stage 5/8] Saving model checkpoint …")
    save_model_checkpoint(model, ckpt_dir=MODEL_CKPT_DIR)

    # Stage 6 — Batch correction
    logger.info("[Stage 6/8] Applying batch correction (zero-OH decoding) …")
    corrected = run_correction(model, raw_embs, batch_idx)
    joint.obsm[CAE_OBSM_KEY] = corrected
    logger.info(
        f"cAE embedding stored at .obsm['{CAE_OBSM_KEY}'] | "
        f"shape={corrected.shape}"
    )

    # Stage 7 — Projections
    logger.info("[Stage 7/8] Computing projections on corrected embeddings …")

    umap_coords = run_umap_3d(corrected, seed=_SEED)
    joint.obsm[UMAP_OBSM_KEY] = umap_coords
    logger.info(
        f"UMAP stored at .obsm['{UMAP_OBSM_KEY}'] | shape={umap_coords.shape}"
    )

    pca_coords = run_pca(corrected, n_components=PCA_N_COMPONENTS, seed=_SEED)
    joint.obsm[PCA_OBSM_KEY] = pca_coords
    logger.info(
        f"PCA stored at .obsm['{PCA_OBSM_KEY}'] | shape={pca_coords.shape}"
    )

    # Stage 8 — Save AnnData
    logger.info("[Stage 8/8] Saving joint corrected AnnData …")
    logger.info("Final .obsm contents:")
    for key, arr in joint.obsm.items():
        logger.info(f"  .obsm['{key}']  shape={np.asarray(arr).shape}")

    save_result(joint, OUTPUT_PATH)

    total = time.perf_counter() - wall_start
    logger.info("=" * 65)
    logger.info(f"Pipeline complete in {total:.1f}s  ({total / 60:.1f} min)")
    logger.info(f"Output:     {OUTPUT_PATH.resolve()}")
    logger.info(f"Checkpoint: {(MODEL_CKPT_DIR / 'cae_trained.pt').resolve()}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()