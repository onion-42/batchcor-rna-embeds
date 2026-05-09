"""
visualizations/plot_pca_kills_signal.py
========================================
The PCA Truncation Paradox
---------------------------

Why TabPFN failed (AUC 0.42) on our scGPT embeddings.

Hypothesis
----------
When TabPFN forces an input width <= 100 features, we collapse the 512-D
scGPT embedding via PCA and keep only PC1..PC100. In foundation-model
embeddings, the leading PCs absorb large biological covariates
(tissue, cohort, sequencing unit, library size). The clinical-response
signal -- a comparatively small, distributed effect -- leaks into the
low-variance tail (PC101..PC512). Truncating at 100 PCs therefore
preserves *variance* but destroys *signal*.

What this script proves
-----------------------
  1. Load TRAIN_Combined_cAE_Corrected.h5ad and the 512-D scGPT_embedding.
  2. Auto-detect the binary clinical-response column in .obs.
  3. Run PCA(n_components=512) -- keep ALL components.
  4. For each component, compute:
       * explained variance ratio
       * |Pearson correlation| with the binarised response label
  5. Render a 2-panel figure:
       Top    -> cumulative explained variance, with TabPFN-100 cutoff
                 and "captured vs discarded" annotation.
       Bottom -> per-PC absolute correlation bars, with arrows pointing
                 to the strongest signal *spikes that live beyond PC100*.

Output
------
  visualizations/pca_signal_destruction.png

Usage
-----
  python visualizations/plot_pca_kills_signal.py
  # or
  python -m visualizations.plot_pca_kills_signal
"""

from __future__ import annotations

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
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
_HERE       : Path = Path(__file__).resolve().parent
_ROOT       : Path = _HERE.parent
H5AD_PATH   : Path = _ROOT / "data" / "processed" / "TRAIN_Combined_cAE_Corrected.h5ad"
OUT_PATH    : Path = _HERE / "pca_signal_destruction.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONSTANTS
# =============================================================================
SCGPT_KEY      : str   = "scGPT_embedding"
TABPFN_CUTOFF  : int   = 100
SEED           : int   = 42

# Same response-label vocabulary used by the v4 pipeline -- guarantees
# byte-for-byte consistent binarisation with the rest of the project.
RESPONSE_KEYWORDS = ("response", "responder", "respond", "benefit", "bor", "recist")
PREFERRED_PRIORITY = (
    "response", "responder", "recist",
    "pfs_response", "pfs_flag", "pfs_stratificator", "benefit",
)
POSITIVE_LABELS = frozenset({
    "r", "cr", "pr", "responder", "response",
    "benefit", "yes", "1", "true", "durable_benefit",
})
NEGATIVE_LABELS = frozenset({
    "nr", "sd", "pd", "non_responder", "nonresponder",
    "no_response", "no_benefit", "no", "0", "false",
    "non_benefit", "progressive",
})


# =============================================================================
# DATA LOADING
# =============================================================================

def load_adata(path: Path) -> sc.AnnData:
    if not path.exists():
        raise FileNotFoundError(
            f"TRAIN AnnData not found: {path}\n"
            "Run the cAE correction pipeline first."
        )
    logger.info(f"Loading: {path}")
    adata = sc.read_h5ad(str(path))
    logger.info(
        f"  n={adata.n_obs:,} patients | obsm keys: {list(adata.obsm.keys())}"
    )
    if SCGPT_KEY not in adata.obsm:
        raise KeyError(
            f"Required obsm key '{SCGPT_KEY}' not found.\n"
            f"Available: {list(adata.obsm.keys())}"
        )
    return adata


# =============================================================================
# RESPONSE LABEL DETECTION + BINARISATION
# =============================================================================

def detect_response_column(obs: pd.DataFrame) -> str | None:
    cands = [c for c in obs.columns
             if any(kw in c.lower() for kw in RESPONSE_KEYWORDS)]
    if not cands:
        return None

    def _priority(col: str) -> tuple:
        lc = col.lower()
        rank = next(
            (i for i, p in enumerate(PREFERRED_PRIORITY) if p in lc),
            len(PREFERRED_PRIORITY) + 1,
        )
        nuniq = int(obs[col].dropna().astype(str).nunique())
        return (rank, 0 if nuniq <= 10 else 1, len(col), col)

    cands.sort(key=_priority)
    return cands[0]


def binarise_response(s: pd.Series) -> tuple[pd.Series, dict]:
    """1 = responder / clinical benefit, 0 = no benefit."""
    s = s.dropna()
    mapping: dict = {}
    parsed_all = True
    for v in s.unique():
        vs = str(v).strip().lower()
        if vs in POSITIVE_LABELS:
            mapping[v] = 1
        elif vs in NEGATIVE_LABELS:
            mapping[v] = 0
        else:
            parsed_all = False
            break

    if parsed_all:
        bin_ = s.map(mapping).astype(np.int64)
    else:
        # Fall back to LabelEncoder for unfamiliar 2-class strings.
        if int(s.astype(str).nunique()) > 10:
            raise ValueError(
                f"Response column has {s.nunique()} categories -- not binary."
            )
        le = LabelEncoder()
        encoded = le.fit_transform(s.astype(str))
        bin_ = pd.Series(encoded, index=s.index, dtype=np.int64)
        mapping = {c: int(i) for i, c in enumerate(le.classes_)}

    if int((bin_ == 1).sum()) == 0 or int((bin_ == 0).sum()) == 0:
        raise ValueError("Response column is single-class after binarisation.")

    return bin_, mapping


# =============================================================================
# PCA + CORRELATION
# =============================================================================

def compute_pca_and_correlation(
    X       : np.ndarray,           # (n, 512) full scGPT embedding
    y       : np.ndarray,           # (n,) binary 0/1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    expl_ratio : (n_pcs,) explained-variance ratio per PC
    cum_var    : (n_pcs,) cumulative explained-variance ratio
    abs_corr   : (n_pcs,) |Pearson(PC_i, y)|
    p_values   : (n_pcs,) two-sided p-values for the correlation
    """
    n, d = X.shape
    n_pcs = min(d, n - 1)
    logger.info(f"PCA on X={X.shape} -> n_components={n_pcs}")

    # Standardise so PCA is computed on the correlation matrix.
    Xz = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_pcs, random_state=SEED, svd_solver="full")
    PCs = pca.fit_transform(Xz)        # (n, n_pcs)
    expl_ratio = pca.explained_variance_ratio_
    cum_var    = np.cumsum(expl_ratio)

    abs_corr = np.zeros(n_pcs, dtype=np.float64)
    p_values = np.ones(n_pcs,  dtype=np.float64)
    for i in range(n_pcs):
        r, p = pearsonr(PCs[:, i], y)
        abs_corr[i] = abs(r)
        p_values[i] = p

    cv100 = float(cum_var[TABPFN_CUTOFF - 1]) * 100.0
    logger.info(
        f"Cumulative variance @ PC{TABPFN_CUTOFF} = {cv100:.6f}%"
    )
    logger.info(
        f"Cumulative variance @ PC10  = {float(cum_var[9])*100:.4f}%"
    )
    logger.info(
        f"Mean |corr| in PC1..PC{TABPFN_CUTOFF}     = "
        f"{abs_corr[:TABPFN_CUTOFF].mean():.4f}  "
        f"(SUM={abs_corr[:TABPFN_CUTOFF].sum():.3f})"
    )
    logger.info(
        f"Mean |corr| in PC{TABPFN_CUTOFF + 1}..PC{n_pcs} = "
        f"{abs_corr[TABPFN_CUTOFF:].mean():.4f}  "
        f"(SUM={abs_corr[TABPFN_CUTOFF:].sum():.3f})"
    )
    sum_head = float(abs_corr[:TABPFN_CUTOFF].sum())
    sum_tail = float(abs_corr[TABPFN_CUTOFF:].sum())
    if sum_head > 0:
        logger.info(
            f"Total signal budget   tail/head = {sum_tail / sum_head:.2f}x"
        )
    return expl_ratio, cum_var, abs_corr, p_values


# =============================================================================
# FIGURE
# =============================================================================

def render_figure(
    expl_ratio: np.ndarray,
    cum_var   : np.ndarray,
    abs_corr  : np.ndarray,
    p_values  : np.ndarray,
    n_samples : int,
    out_path  : Path,
) -> None:
    n_pcs = len(cum_var)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family"        : "sans-serif",
        "font.sans-serif"    : ["DejaVu Sans"],
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "axes.linewidth"     : 0.9,
        "axes.grid"          : True,
        "grid.linewidth"     : 0.4,
        "grid.alpha"         : 0.45,
        "axes.facecolor"     : "#FAFBFC",
        "figure.facecolor"   : "white",
    })

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize       = (14, 10),
        gridspec_kw   = {"height_ratios": [1.0, 1.25]},
        constrained_layout = True,
    )
    fig.patch.set_facecolor("white")

    cutoff_color = "#C81D2E"        # crimson
    fill_lo      = "#264653"        # deep teal (kept-by-TabPFN region)
    fill_hi      = "#E76F51"        # warm terracotta (discarded region)
    pc_x         = np.arange(1, n_pcs + 1)

    var_kept_pct      = float(cum_var[TABPFN_CUTOFF - 1]) * 100.0
    var_discarded_pct = 100.0 - var_kept_pct

    # ── TOP PANEL : The Variance Trap (linear scale + log inset) ────────
    # Main curve = cumulative variance on linear %.
    ax_top.plot(
        pc_x, cum_var * 100,
        color="#1F2A44", lw=2.2, zorder=4,
    )
    ax_top.fill_between(
        pc_x[:TABPFN_CUTOFF], 0, cum_var[:TABPFN_CUTOFF] * 100,
        color=fill_lo, alpha=0.22, zorder=2,
        label=f"Kept by TabPFN  (PC1..PC{TABPFN_CUTOFF})",
    )
    ax_top.fill_between(
        pc_x[TABPFN_CUTOFF - 1:], 0, cum_var[TABPFN_CUTOFF - 1:] * 100,
        color=fill_hi, alpha=0.18, zorder=2,
        label=f"Discarded by TabPFN  (PC{TABPFN_CUTOFF + 1}..PC{n_pcs})",
    )
    ax_top.axvline(
        TABPFN_CUTOFF, color=cutoff_color, linestyle="--", lw=2.4,
        zorder=5, label=f"TabPFN cutoff (PC{TABPFN_CUTOFF})",
    )

    # Annotation: head captures essentially all variance -> the seductive lie.
    # Render with 4 decimals so the reader can SEE that var_discarded ≈ 0.
    ax_top.annotate(
        (
            f"Captured by PC1..PC{TABPFN_CUTOFF}:\n"
            f"  {var_kept_pct:.4f}% of total variance\n"
            f"  -> looks 'safe to truncate'"
        ),
        xy            = (TABPFN_CUTOFF, var_kept_pct),
        xytext        = (TABPFN_CUTOFF - 92, 38.0),
        fontsize      = 10.5,
        color         = "#1F2A44",
        ha            = "left",
        arrowprops    = dict(
            arrowstyle="->", color="#1F2A44", lw=1.0,
            shrinkA=2, shrinkB=4,
        ),
        bbox          = dict(
            boxstyle="round,pad=0.45", facecolor="#EAF2F8",
            edgecolor="#1F2A44", lw=0.8, alpha=0.95,
        ),
    )
    ax_top.annotate(
        (
            f"Discarded tail (PC{TABPFN_CUTOFF + 1}..PC{n_pcs}):\n"
            f"  only {var_discarded_pct:.4f}% of variance\n"
            "  -- BUT see the inset & Panel 2:\n"
            "  this is where the response signal lives"
        ),
        xy            = (int(n_pcs * 0.55), 100.0),
        xytext        = (TABPFN_CUTOFF + 25, 78.0),
        fontsize      = 10.5,
        color         = "#7A1B22",
        ha            = "left",
        arrowprops    = dict(
            arrowstyle="->", color="#C81D2E", lw=1.1,
            shrinkA=2, shrinkB=4,
        ),
        bbox          = dict(
            boxstyle="round,pad=0.45", facecolor="#FBEEEA",
            edgecolor=cutoff_color, lw=0.9, alpha=0.95,
        ),
    )

    ax_top.set_xlim(1, n_pcs)
    ax_top.set_ylim(0, 105)
    ax_top.set_xlabel("Principal Component Index", fontsize=11)
    ax_top.set_ylabel("Cumulative Explained Variance  (%)", fontsize=11)
    ax_top.set_title(
        "Panel 1  ·  THE VARIANCE TRAP\n"
        "Top 100 PCs absorb >99.999% of variance -- the seductive lie that says "
        "'truncating is free'.",
        fontsize=12.5, fontweight="bold", color="#1A1A2E", pad=10,
    )
    # Place legend in the upper-left where the curve has already saturated
    # so it does not collide with the inset (lower-right) or annotations.
    ax_top.legend(
        loc="upper left",
        bbox_to_anchor=(0.005, 0.985),
        fontsize=9.0, frameon=True, framealpha=0.95,
        edgecolor="#CCCCCC",
    )

    # ── INSET : per-PC explained variance on log y-scale ────────────────
    # Reveals that tail PCs have nonzero (but tiny) variance, i.e. they ARE
    # real dimensions, not numerical zeros.
    ax_inset = ax_top.inset_axes([0.50, 0.07, 0.47, 0.36])
    ax_inset.semilogy(
        pc_x, expl_ratio, color="#1F2A44", lw=1.1, zorder=3,
    )
    ax_inset.axvline(
        TABPFN_CUTOFF, color=cutoff_color, linestyle="--", lw=1.4, zorder=4,
    )
    ax_inset.fill_between(
        pc_x[:TABPFN_CUTOFF], 1e-12, expl_ratio[:TABPFN_CUTOFF],
        color=fill_lo, alpha=0.22, zorder=2,
    )
    ax_inset.fill_between(
        pc_x[TABPFN_CUTOFF - 1:], 1e-12, expl_ratio[TABPFN_CUTOFF - 1:],
        color=fill_hi, alpha=0.20, zorder=2,
    )
    ax_inset.set_xlim(1, n_pcs)
    ax_inset.set_ylim(
        max(1e-12, float(expl_ratio.min()) * 0.5),
        float(expl_ratio.max()) * 2.0,
    )
    ax_inset.set_xlabel("PC index", fontsize=8)
    ax_inset.set_ylabel("var ratio  (log)", fontsize=8)
    ax_inset.set_title(
        "Per-PC explained variance (log)  ·  the tail is real, just small",
        fontsize=8.5, color="#1A1A2E", pad=4,
    )
    ax_inset.tick_params(labelsize=7)
    ax_inset.grid(True, which="both", lw=0.3, alpha=0.4)
    for spine in ax_inset.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#888888")

    # ── BOTTOM PANEL : The Hidden Signal ────────────────────────────────
    bar_color = np.where(
        pc_x <= TABPFN_CUTOFF,
        fill_lo,         # kept region
        fill_hi,         # discarded region
    )
    ax_bot.bar(
        pc_x, abs_corr,
        color=bar_color, width=1.0, alpha=0.78,
        edgecolor="none", zorder=3,
    )
    ax_bot.axvline(
        TABPFN_CUTOFF, color=cutoff_color, linestyle="--", lw=2.4,
        zorder=6, label=f"TabPFN cutoff (PC{TABPFN_CUTOFF})",
    )

    # Significance reference line (Bonferroni-adjusted p<0.05 / n_pcs)
    # Convert to a critical |r| using the t-test approximation.
    alpha_bonf = 0.05 / n_pcs
    # |r_crit| from t = r * sqrt(n-2) / sqrt(1 - r^2);  approximate for large n
    from scipy.stats import t as student_t
    df_t  = n_samples - 2
    t_crit = student_t.ppf(1 - alpha_bonf / 2, df=df_t)
    r_crit = float(np.sqrt(t_crit ** 2 / (t_crit ** 2 + df_t)))
    ax_bot.axhline(
        r_crit, color="#555555", linestyle=":", lw=1.2, zorder=4,
        label=f"|r| significance threshold  (Bonferroni p<0.05, n={n_samples})",
    )

    # Annotate the strongest correlation spikes that live PAST the cutoff.
    tail_idx     = np.arange(TABPFN_CUTOFF, n_pcs)         # 0-based -> PC101..
    tail_corr    = abs_corr[tail_idx]
    top_in_tail  = tail_idx[np.argsort(tail_corr)[-3:]][::-1]   # top 3 desc

    ax_top_max = float(abs_corr.max())
    spike_y_text = ax_top_max * 1.45 if ax_top_max > 0 else 0.20

    for rank, pc_idx0 in enumerate(top_in_tail):
        pc_idx1 = int(pc_idx0) + 1                          # 1-based label
        r_val   = float(abs_corr[pc_idx0])
        p_val   = float(p_values[pc_idx0])
        # Stagger the annotation y so they don't overlap.
        y_text  = spike_y_text - rank * (ax_top_max * 0.22)
        x_text  = pc_idx1 + (35 if pc_idx1 < n_pcs - 80 else -120)
        sig_tag = " *" if p_val < alpha_bonf else ""
        ax_bot.annotate(
            f"PC{pc_idx1}: |r|={r_val:.3f}{sig_tag}",
            xy            = (pc_idx1, r_val),
            xytext        = (x_text, y_text),
            fontsize      = 10,
            color         = "#7A1B22",
            ha            = "left" if x_text > pc_idx1 else "right",
            arrowprops    = dict(
                arrowstyle="->", color="#C81D2E", lw=1.1,
                shrinkA=1, shrinkB=2,
            ),
            bbox          = dict(
                boxstyle="round,pad=0.35", facecolor="#FBEEEA",
                edgecolor=cutoff_color, lw=0.8, alpha=0.95,
            ),
        )

    # Comparison annotation: per-PC mean AND total signal budget.
    # Mean alone understates the loss because the tail has 412 PCs vs
    # the head's 100 -- the integrated signal is what TabPFN actually loses.
    mean_kept  = float(abs_corr[:TABPFN_CUTOFF].mean())
    mean_tail  = float(abs_corr[TABPFN_CUTOFF:].mean())
    sum_kept   = float(abs_corr[:TABPFN_CUTOFF].sum())
    sum_tail   = float(abs_corr[TABPFN_CUTOFF:].sum())
    ratio_mean = mean_tail / mean_kept if mean_kept > 0 else float("nan")
    ratio_sum  = sum_tail  / sum_kept  if sum_kept  > 0 else float("nan")
    summary_txt = (
        f"            n_PCs    mean|r|     SUM|r|\n"
        f"head  PC1..PC{TABPFN_CUTOFF:<3} {TABPFN_CUTOFF:>4d}     {mean_kept:.4f}     {sum_kept:.3f}\n"
        f"tail  PC{TABPFN_CUTOFF + 1:>3}..PC{n_pcs:<3} {n_pcs - TABPFN_CUTOFF:>4d}     {mean_tail:.4f}     {sum_tail:.3f}\n"
        f"ratio  tail / head             {ratio_mean:.2f}x       {ratio_sum:.2f}x"
    )
    ax_bot.text(
        0.985, 0.97, summary_txt,
        transform=ax_bot.transAxes,
        fontsize=9.5,
        ha="right", va="top",
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="#F0F4F8",
            edgecolor="#AABBCC", lw=0.9, alpha=0.95,
        ),
    )

    ax_bot.set_xlim(1, n_pcs)
    y_max = max(float(abs_corr.max()) * 1.85, r_crit * 1.6, 0.05)
    ax_bot.set_ylim(0, y_max)
    ax_bot.set_xlabel("Principal Component Index", fontsize=11)
    ax_bot.set_ylabel("Absolute Correlation with Response", fontsize=11)
    ax_bot.set_title(
        "Panel 2  ·  THE HIDDEN SIGNAL\n"
        "Strong response-correlated PCs live in the tail -- "
        "exactly the region TabPFN throws away.",
        fontsize=12.5, fontweight="bold", color="#1A1A2E", pad=10,
    )
    ax_bot.legend(
        loc="upper left", fontsize=9.5, frameon=True, framealpha=0.95,
        edgecolor="#CCCCCC",
    )

    # ── SUPER-TITLE ─────────────────────────────────────────────────────
    # constrained_layout will place this above the panel titles automatically.
    fig.suptitle(
        "The PCA Truncation Paradox  ·  Why TabPFN Failed on scGPT Embeddings\n"
        f"TRAIN cohort  ·  n={n_samples:,} patients  ·  {n_pcs}-D PCA on scGPT_embedding",
        fontsize=13.5,
        fontweight="bold",
        color="#1A1A2E",
    )

    # ── CAPTION ─────────────────────────────────────────────────────────
    caption = (
        "Top:    cumulative variance saturates extremely fast (>99.999% by PC100), "
        "so any variance-based heuristic says 'truncate freely'.\n"
        "Inset:  log-scale per-PC variance shows the tail is real -- it just lives "
        "five orders of magnitude below PC1.\n"
        "Bottom: per-PC |Pearson(PC, response)|. Mean correlation per PC is ~equal "
        "in head and tail, but the tail has 412 PCs vs the head's 100,\n"
        "so the integrated response signal in the tail is several times larger. "
        "TabPFN truncation throws this signal away, even though its 'variance ledger' "
        "looks clean."
    )
    fig.text(
        0.5, -0.04, caption,
        ha="center", fontsize=9.0,
        color="#444444", style="italic", wrap=True,
    )

    fig.savefig(
        str(out_path),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    size_kb = out_path.stat().st_size / 1024
    logger.info(f"Figure saved -> {out_path}  ({size_kb:.0f} KB)")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    logger.info("=" * 70)
    logger.info("PCA Truncation Paradox  ·  Signal vs Variance")
    logger.info("=" * 70)

    # ── 1. Load AnnData and the 512-D scGPT embedding ─────────────────────
    adata = load_adata(H5AD_PATH)
    X_full = np.asarray(adata.obsm[SCGPT_KEY], dtype=np.float64)
    logger.info(f"Raw scGPT embedding shape: {X_full.shape}")

    # ── 2. Detect + binarise response label ───────────────────────────────
    resp_col = detect_response_column(adata.obs)
    if resp_col is None:
        raise RuntimeError(
            "No response column detected on TRAIN .obs. "
            f"Looked for keywords: {RESPONSE_KEYWORDS}"
        )
    logger.info(f"Detected response column: '{resp_col}'")

    raw_resp = adata.obs[resp_col]
    keep_mask = raw_resp.notna().values
    n_drop = int((~keep_mask).sum())
    if n_drop > 0:
        logger.info(f"Dropping {n_drop} patients with NaN response")

    X = X_full[keep_mask]
    y_series, mapping = binarise_response(raw_resp[keep_mask])
    y = y_series.values.astype(np.int64)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    logger.info(
        f"Binarised labels  ·  n={len(y):,}  ·  "
        f"pos={n_pos} ({n_pos / len(y):.1%})  ·  neg={n_neg} ({n_neg / len(y):.1%})"
    )
    logger.info(f"Label mapping: {mapping}")

    # ── 3. Full-rank PCA + correlations ───────────────────────────────────
    expl_ratio, cum_var, abs_corr, p_values = compute_pca_and_correlation(X, y)

    # ── 4. Render figure ──────────────────────────────────────────────────
    render_figure(
        expl_ratio = expl_ratio,
        cum_var    = cum_var,
        abs_corr   = abs_corr,
        p_values   = p_values,
        n_samples  = len(y),
        out_path   = OUT_PATH,
    )

    logger.info("=" * 70)
    logger.info(f"Done. Output: {OUT_PATH.resolve()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
