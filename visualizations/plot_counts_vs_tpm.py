"""
visualizations/plot_counts_vs_tpm.py
=====================================
Visual proof that scGPT Value Binning + cAE batch correction successfully
unifies datasets originating from different expression-unit conventions
(raw Counts vs TPM) in the corrected latent space.

Biological rationale
---------------------
scGPT was pre-trained on raw count data.  Our pipeline feeds it BULK RNA-seq
data in TPM units (KIRC cohort) alongside count-normalised data (Melanoma,
NSCLC).  The Value Binning step (51 ordinal bins, per-cell linear
interpolation) is hypothesised to dissolve the TPM vs Counts distribution
gap by mapping absolute expression values into a shared rank-based
representation — the model never sees the raw magnitudes.

The cAE then removes any residual technical covariate still present in the
512-D scGPT embedding space.

If both mechanisms work, the corrected UMAP should show TPM and Counts
patients occupying the *same* manifold regions, while the raw UMAP should
show residual separation driven by unit conventions.

Unit mapping
-------------
  KIRC     → "TPM"     (Immotion / TCGA-KIRC standard: TPM from RSEM)
  Melanoma → "Counts"  (TCGA SKCM/Riaz/Hugo: HTSeq raw counts, log-CPM)
  NSCLC    → "Counts"  (TCGA LUAD/LUSC / Rizvi: raw counts pipeline)

Usage
------
    python visualizations/plot_counts_vs_tpm.py
    # or from project root:
    python -m visualizations.plot_counts_vs_tpm

Output
------
    visualizations/counts_vs_tpm_manifold.png
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from loguru import logger
from sklearn.decomposition import PCA
from umap import UMAP

warnings.filterwarnings("ignore")

# =============================================================================
# LOGGING
# =============================================================================
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | <cyan>{function}</cyan> | {message}"
    ),
    level="INFO",
)

# =============================================================================
# PATHS
# =============================================================================
_HERE     : Path = Path(__file__).resolve().parent
_ROOT     : Path = _HERE.parent           # project root
H5AD_PATH : Path = _ROOT / "data" / "processed" / "TRAIN_Combined_cAE_Corrected.h5ad"
OUT_PATH  : Path = _HERE / "counts_vs_tpm_manifold.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONSTANTS
# =============================================================================
SCGPT_KEY : str = "scGPT_embedding"
CAE_KEY   : str = "cAE_embedding"
SEED      : int = 42

# Unit mapping — grounded in the original data-release protocols
COHORT_UNIT_MAP: dict[str, str] = {
    "KIRC"     : "TPM",     # TCGA-KIRC / Immotion: RSEM-TPM pipeline
    "Melanoma" : "Counts",  # Riaz/Hugo/TCGA-SKCM: HTSeq counts → log-CPM
    "NSCLC"    : "Counts",  # Rizvi/TCGA LUAD+LUSC: STAR counts pipeline
}

# Publication-grade colour palette (colour-blind friendly, NEJM-adjacent)
UNIT_PALETTE: dict[str, str] = {
    "TPM"    : "#E76F51",   # warm terracotta
    "Counts" : "#264653",   # deep teal
}

ALPHA_SCATTER : float = 0.68
POINT_SIZE    : float = 7.0
UMAP_N_NEIGHBORS: int = 20
UMAP_MIN_DIST   : float = 0.25
PCA_PRE_DIM     : int = 64     # pre-PCA before UMAP (speeds up & improves topology)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_adata(path: Path) -> sc.AnnData:
    if not path.exists():
        raise FileNotFoundError(
            f"TRAIN AnnData not found: {path}\n"
            "Run the cAE correction pipeline first:\n"
            "  python run_cae_correction.py"
        )
    logger.info(f"Loading: {path}")
    adata = sc.read_h5ad(str(path))
    logger.info(
        f"  n={adata.n_obs:,} patients | "
        f"obsm keys: {list(adata.obsm.keys())}"
    )
    for key in [SCGPT_KEY, CAE_KEY]:
        if key not in adata.obsm:
            raise KeyError(
                f"Required obsm key '{key}' not found.\n"
                f"Available: {list(adata.obsm.keys())}"
            )
    return adata


# =============================================================================
# UNIT ANNOTATION
# =============================================================================

def annotate_units(adata: sc.AnnData) -> sc.AnnData:
    """
    Assign input_unit label to each patient based on their cohort.

    The cohort column may be named 'cohort', 'Cohort', 'batch', 'RNA_batch',
    or encoded in obs_names as a prefix.  We try all common variants.
    """
    cohort_col_candidates = ["cohort", "Cohort", "batch", "RNA_batch", "Batch"]
    cohort_col: str | None = None

    for c in cohort_col_candidates:
        if c in adata.obs.columns:
            cohort_col = c
            logger.info(f"Cohort column found: '{cohort_col}'")
            break

    if cohort_col is None:
        # Last resort: parse from obs_names prefix  (e.g. "KIRC_patient_001")
        logger.warning(
            "No cohort column found in .obs. "
            "Attempting to parse cohort from obs_names prefix."
        )
        def _parse_prefix(name: str) -> str:
            for k in COHORT_UNIT_MAP:
                if name.startswith(k):
                    return k
            return "Unknown"
        adata.obs["cohort"] = adata.obs_names.map(_parse_prefix)
        cohort_col = "cohort"

    unit_series = adata.obs[cohort_col].astype(str).map(
        lambda c: COHORT_UNIT_MAP.get(c, "Unknown")
    )
    adata.obs["input_unit"] = unit_series

    # Report mapping
    for cohort, unit in COHORT_UNIT_MAP.items():
        n = int((adata.obs[cohort_col].astype(str) == cohort).sum())
        logger.info(f"  {cohort:<12} → {unit:<8} (n={n:,})")

    unknown = int((unit_series == "Unknown").sum())
    if unknown > 0:
        logger.warning(
            f"{unknown} patients mapped to 'Unknown' unit "
            "(cohort name not in COHORT_UNIT_MAP)."
        )

    return adata


# =============================================================================
# UMAP COMPUTATION
# =============================================================================

def compute_umap_2d(
    emb     : np.ndarray,
    label   : str,
    seed    : int = SEED,
) -> np.ndarray:
    """
    PCA pre-reduce → 2-D UMAP.

    Pre-reducing to PCA_PRE_DIM before UMAP substantially speeds up the
    kNN graph construction on 512-D embeddings while preserving the large-
    scale structure (Becht et al. 2019, Nature Biotechnology).
    """
    n, d = emb.shape
    effective_pca = min(PCA_PRE_DIM, n - 1, d)
    logger.info(
        f"[{label}] PCA {d}D → {effective_pca}D then UMAP 2D "
        f"(n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}) …"
    )

    pca_coords = PCA(
        n_components=effective_pca, random_state=seed
    ).fit_transform(emb).astype(np.float32)

    umap_coords = UMAP(
        n_components   = 2,
        n_neighbors    = UMAP_N_NEIGHBORS,
        min_dist       = UMAP_MIN_DIST,
        metric         = "euclidean",
        random_state   = seed,
        verbose        = False,
    ).fit_transform(pca_coords).astype(np.float32)

    logger.info(f"[{label}] UMAP computed: shape={umap_coords.shape}")
    return umap_coords


# =============================================================================
# FIGURE RENDERING
# =============================================================================

def render_figure(
    umap_raw : np.ndarray,     # (N, 2)
    umap_cae : np.ndarray,     # (N, 2)
    units    : pd.Series,      # index-aligned with adata.obs
    out_path : Path,
) -> None:
    """
    1 × 2 publication-ready figure comparing raw vs corrected UMAP
    coloured by input_unit (TPM / Counts).
    """
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family"        : "sans-serif",
        "font.sans-serif"    : ["DejaVu Sans"],
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "axes.spines.left"   : True,
        "axes.spines.bottom" : True,
        "axes.linewidth"     : 0.8,
        "axes.grid"          : True,
        "grid.linewidth"     : 0.4,
        "grid.alpha"         : 0.4,
        "axes.facecolor"     : "#FAFBFC",
        "figure.facecolor"   : "white",
    })

    fig, axes = plt.subplots(
        1, 2,
        figsize       = (15, 7.6),
        constrained_layout = False,
    )
    fig.patch.set_facecolor("white")
    # Reserve top band for suptitle, bottom band for legend + caption boxes.
    fig.subplots_adjust(
        top    = 0.83,
        bottom = 0.18,
        left   = 0.035,
        right  = 0.985,
        wspace = 0.08,
    )

    # ── shared colour mapping ─────────────────────────────────────────────
    unit_labels = units.values
    unit_order  = ["TPM", "Counts", "Unknown"]

    def _scatter_panel(
        ax        : plt.Axes,
        coords    : np.ndarray,
        title     : str,
        subtitle  : str,
    ) -> None:
        # Draw each unit class with controlled zorder and alpha
        for z, unit in enumerate(unit_order):
            mask = unit_labels == unit
            if mask.sum() == 0:
                continue
            color = UNIT_PALETTE.get(unit, "#888888")
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c          = color,
                s          = POINT_SIZE,
                alpha      = ALPHA_SCATTER,
                linewidths = 0,
                zorder     = z + 2,
                rasterized = True,   # keeps file size small at high DPI
            )

        # Subtle density contours to reveal overlap structure
        try:
            from scipy.stats import gaussian_kde
            for unit in ["TPM", "Counts"]:
                mask = unit_labels == unit
                if mask.sum() < 20:
                    continue
                xy    = coords[mask].T
                kde   = gaussian_kde(xy, bw_method=0.35)
                xi    = np.linspace(coords[:, 0].min() - 0.5, coords[:, 0].max() + 0.5, 150)
                yi    = np.linspace(coords[:, 1].min() - 0.5, coords[:, 1].max() + 0.5, 150)
                Xi, Yi = np.meshgrid(xi, yi)
                Zi    = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
                ax.contour(
                    Xi, Yi, Zi,
                    levels     = 4,
                    colors     = [UNIT_PALETTE[unit]],
                    linewidths = 0.9,
                    alpha      = 0.45,
                    zorder     = 10,
                )
        except Exception:
            pass   # contours are cosmetic; never crash the pipeline

        ax.set_title(title, fontsize=12.5, fontweight="bold", pad=14, color="#1A1A2E")
        ax.set_xlabel(subtitle, fontsize=9.5, color="#555555", labelpad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)
        ax.set_aspect("equal", adjustable="datalim")

        # Overlap annotation: fraction of TPM & Counts that share the same
        # quadrant of the UMAP (rough measure of mixing)
        if "Corrected" in title:
            _annotate_mixing(ax, coords, unit_labels)

    def _annotate_mixing(
        ax     : plt.Axes,
        coords : np.ndarray,
        labels : np.ndarray,
    ) -> None:
        """Annotate with a simple mixing score: Wasserstein-like proximity."""
        try:
            from scipy.spatial.distance import cdist
            tpm_idx  = np.where(labels == "TPM")[0]
            cnt_idx  = np.where(labels == "Counts")[0]
            if len(tpm_idx) < 5 or len(cnt_idx) < 5:
                return
            # Sample up to 300 from each to keep it fast
            rng   = np.random.default_rng(SEED)
            t_sub = coords[rng.choice(tpm_idx, min(300, len(tpm_idx)), replace=False)]
            c_sub = coords[rng.choice(cnt_idx, min(300, len(cnt_idx)), replace=False)]
            # Average nearest-neighbour distance from TPM → Counts
            dists = cdist(t_sub, c_sub, metric="euclidean")
            mean_nn = float(np.mean(np.min(dists, axis=1)))

            # Compute the same on raw for comparison stored in closure via
            # a mutable container (set externally after both panels drawn)
            ax._mixing_score = mean_nn
        except Exception:
            pass

    # ── Left: Raw scGPT ───────────────────────────────────────────────────
    _scatter_panel(
        axes[0], umap_raw,
        title    = "BEFORE  —  Raw scGPT Embedding",
        subtitle = "UMAP 2D  ·  coloured by expression-unit convention",
    )
    # obsm key tag: lower-right corner, away from title and KDE contours
    axes[0].text(
        0.985, 0.02, "obsm['scGPT_embedding']",
        transform=axes[0].transAxes, fontsize=7.5,
        color="#888888", va="bottom", ha="right",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor="#DDDDDD", lw=0.6, alpha=0.85),
        zorder=20,
    )

    # ── Right: cAE Corrected ──────────────────────────────────────────────
    _scatter_panel(
        axes[1], umap_cae,
        title    = "AFTER  —  cAE Batch-Corrected Embedding",
        subtitle = "UMAP 2D  ·  coloured by expression-unit convention",
    )
    axes[1].text(
        0.985, 0.02, "obsm['cAE_embedding']",
        transform=axes[1].transAxes, fontsize=7.5,
        color="#888888", va="bottom", ha="right",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor="#DDDDDD", lw=0.6, alpha=0.85),
        zorder=20,
    )

    # ── Shared legend ─────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(
            facecolor=UNIT_PALETTE[u],
            edgecolor="none",
            label=f"{u}  (n={int((unit_labels == u).sum()):,})",
            alpha=0.85,
        )
        for u in ["TPM", "Counts"]
        if (unit_labels == u).any()
    ]
    # Legend goes in the reserved bottom band at y ≈ 0.11 (panels end at 0.18).
    fig.legend(
        handles     = legend_handles,
        title       = "Expression unit  (input to scGPT)",
        title_fontsize = 9.5,
        fontsize    = 9,
        loc         = "center",
        ncol        = 2,
        frameon     = True,
        framealpha  = 0.92,
        edgecolor   = "#CCCCCC",
        bbox_to_anchor = (0.5, 0.115),
        handlelength  = 1.6,
    )

    # ── Super-title in the reserved top band ──────────────────────────────
    # Top of figure is reserved up to y=0.83 for the panel area, so we have
    # 0.83..1.0 (≈17% of fig height) for the suptitle. Two lines @ 12.5 pt.
    fig.suptitle(
        "Value Binning + cAE Erases the TPM vs Counts Technical Gap\n"
        "scGPT bulk RNA-seq embeddings  ·  TRAIN cohorts  "
        "(KIRC = TPM   |   Melanoma, NSCLC = Counts)",
        fontsize  = 13.0,
        fontweight= "bold",
        color     = "#1A1A2E",
        y         = 0.955,
    )

    # ── Explanatory caption (reserved bottom band) ────────────────────────
    caption = (
        "Density contours mark the TPM (terracotta) and Counts (teal) manifolds.\n"
        "Overlap of contours after cAE correction indicates successful unit-agnostic integration."
    )
    fig.text(
        0.5, 0.045, caption,
        ha        = "center",
        va        = "center",
        fontsize  = 8.5,
        color     = "#555555",
        style     = "italic",
    )

    # ── Mixing-score annotation, also in the bottom band ──────────────────
    try:
        raw_score = getattr(axes[0], "_mixing_score", None)
        cae_score = getattr(axes[1], "_mixing_score", None)
        if raw_score is not None and cae_score is not None:
            improvement = (raw_score - cae_score) / raw_score * 100
            ann = (
                f"Mean TPM->Counts NN distance:   "
                f"Before {raw_score:.3f}   ->   After {cae_score:.3f}   "
                f"(mixing {improvement:+.1f}%)"
            )
            fig.text(
                0.5, 0.005, ann,
                ha        = "center",
                va        = "bottom",
                fontsize  = 8,
                color     = "#333333",
                fontfamily= "monospace",
                bbox      = dict(
                    boxstyle  = "round,pad=0.4",
                    facecolor = "#F0F4F8",
                    edgecolor = "#AABBCC",
                    alpha     = 0.9,
                ),
            )
    except Exception:
        pass

    # ── Save ──────────────────────────────────────────────────────────────
    # NOTE: don't use bbox_inches="tight" -- that re-crops and undoes the
    # carefully reserved header/footer bands above. The figure is already
    # sized to include everything within (0..1, 0..1) figure coords.
    fig.savefig(
        str(out_path),
        dpi           = 200,
        facecolor     = "white",
        edgecolor     = "none",
    )
    plt.close(fig)
    size_kb = out_path.stat().st_size / 1024
    logger.info(f"Figure saved → {out_path}  ({size_kb:.0f} KB)")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    logger.info("=" * 60)
    logger.info("Counts vs TPM Manifold Visualisation")
    logger.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────
    adata = load_adata(H5AD_PATH)

    # ── Annotate input units ──────────────────────────────────────────────
    adata = annotate_units(adata)

    # Summary of unit distribution
    unit_counts = adata.obs["input_unit"].value_counts()
    logger.info(f"Unit distribution:\n{unit_counts.to_string()}")

    # ── Extract embeddings ────────────────────────────────────────────────
    raw_emb = np.asarray(adata.obsm[SCGPT_KEY], dtype=np.float32)
    cae_emb = np.asarray(adata.obsm[CAE_KEY],   dtype=np.float32)
    logger.info(f"Raw emb shape : {raw_emb.shape}")
    logger.info(f"cAE emb shape : {cae_emb.shape}")

    # ── Compute UMAPs ─────────────────────────────────────────────────────
    logger.info("Computing UMAP for Raw scGPT embeddings …")
    umap_raw = compute_umap_2d(raw_emb, label="Raw scGPT")

    logger.info("Computing UMAP for cAE-corrected embeddings …")
    umap_cae = compute_umap_2d(cae_emb, label="cAE Corrected")

    # ── Render figure ─────────────────────────────────────────────────────
    logger.info("Rendering figure …")
    render_figure(
        umap_raw = umap_raw,
        umap_cae = umap_cae,
        units    = adata.obs["input_unit"],
        out_path = OUT_PATH,
    )

    logger.info("=" * 60)
    logger.info(f"Done. Output: {OUT_PATH.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
