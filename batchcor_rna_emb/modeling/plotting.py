"""Publication-quality plotting for downstream ML evaluation.

Generates bar + swarm plots with statistical brackets (paired Wilcoxon
+ Holm correction) comparing pipeline variants (embedding × correction).

Design principles
-----------------
- One plot per (Model, Cohort): all pipelines on x-axis sorted by mean.
- DummyClassifier baseline shown as horizontal dashed line.
- Swarm overlay (black dots) for individual seed scores.
- Adjacent-pair brackets with significance stars.
- Large, readable fonts (publication scale).
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import wilcoxon


# ---------------------------------------------------------------------------
# Significance helpers
# ---------------------------------------------------------------------------

def _sig_stars(p: float) -> str:
    """Convert p-value to significance stars."""
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return "n.s."


def _holm_correction(pvalues: list[float]) -> list[float]:
    """Apply Holm–Bonferroni correction to a list of p-values."""
    n = len(pvalues)
    if n == 0:
        return []

    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    corrected = [0.0] * n

    for rank, (orig_idx, p) in enumerate(indexed):
        corrected[orig_idx] = min(p * (n - rank), 1.0)

    # Enforce monotonicity (each corrected p >= previous in sorted order)
    sorted_indices = [i for i, _ in indexed]
    for k in range(1, n):
        i_curr = sorted_indices[k]
        i_prev = sorted_indices[k - 1]
        corrected[i_curr] = max(corrected[i_curr], corrected[i_prev])

    return corrected


def compute_pairwise_wilcoxon(
    df: pd.DataFrame,
    pipeline_col: str,
    metric_col: str,
    split_col: str = "split",
) -> pd.DataFrame:
    """Compute paired Wilcoxon tests for ALL pairs, apply Holm correction.

    Parameters
    ----------
    df : pd.DataFrame
        Results table with one row per (pipeline, split).
    pipeline_col : str
        Column identifying the pipeline variant.
    metric_col : str
        Column with metric values (e.g. 'f1_weighted').
    split_col : str
        Column identifying the fold/seed.

    Returns
    -------
    pd.DataFrame
        Columns: pipeline_a, pipeline_b, p_raw, p_corrected, stars.
    """
    pipelines = df[pipeline_col].unique()
    pairs = list(combinations(pipelines, 2))
    
    if not pairs:
        return pd.DataFrame(columns=["pipeline_a", "pipeline_b", "p_raw", "p_corrected", "stars"])

    records = []
    for a, b in pairs:
        da = df[df[pipeline_col] == a].set_index(split_col)[metric_col]
        db = df[df[pipeline_col] == b].set_index(split_col)[metric_col]
        common = da.index.intersection(db.index)

        if len(common) < 5:
            records.append({"pipeline_a": a, "pipeline_b": b, "p_raw": 1.0})
            continue

        va = da.loc[common].values
        vb = db.loc[common].values

        # If all differences are zero, Wilcoxon can't compute
        if np.allclose(va, vb):
            records.append({"pipeline_a": a, "pipeline_b": b, "p_raw": 1.0})
            continue

        try:
            _, p = wilcoxon(va, vb, alternative="two-sided")
        except ValueError:
            p = 1.0
        records.append({"pipeline_a": a, "pipeline_b": b, "p_raw": p})

    result = pd.DataFrame(records)
    result["p_corrected"] = _holm_correction(result["p_raw"].tolist())
    result["stars"] = result["p_corrected"].apply(_sig_stars)
    return result


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_pipeline_comparison(
    df: pd.DataFrame,
    metric_col: str = "f1_weighted",
    metric_label: str = "F1 weighted",
    pipeline_col: str = "pipeline",
    split_col: str = "split",
    model_name: str = "",
    cohort_name: str = "",
    n_test: int = 0,
    dummy_score: float | None = None,
    figsize: tuple[float, float] = (10, 7),
    save_path: str | Path | None = None,
    palette: list[str] | None = None,
) -> plt.Figure:
    """Create publication-quality bar + swarm plot with stat brackets.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: ``pipeline_col``, ``split_col``, ``metric_col``.
    metric_col : str
        Which metric to plot on y-axis.
    metric_label : str
        Human-readable label for y-axis.
    pipeline_col : str
        Column identifying the pipeline variant (embedding + correction).
    split_col : str
        Column identifying folds/seeds.
    model_name : str
        ML model name for the title.
    cohort_name : str
        Cohort name for the title.
    n_test : int
        Number of test samples (for the title).
    dummy_score : float or None
        DummyClassifier baseline score. Shown as dashed line.
    figsize : tuple
        Figure size.
    save_path : str or Path or None
        If set, saves figure to this path (PNG, 300 dpi).
    palette : list[str] or None
        Custom color palette. Defaults to a curated set.

    Returns
    -------
    plt.Figure
    """
    if palette is None:
        palette = [
            "#4C72B0", "#DD8452", "#55A868", "#C44E52",
            "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
            "#CCB974", "#64B5CD",
        ]

    # --- Sort pipelines by mean metric ---
    order = (
        df.groupby(pipeline_col)[metric_col]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    # --- Figure setup (large fonts) ---
    fig, ax = plt.subplots(figsize=figsize)

    base_font = 16
    plt.rcParams.update({
        "font.size": base_font,
        "axes.titlesize": base_font + 6,
        "axes.labelsize": base_font + 2,
        "xtick.labelsize": base_font,
        "ytick.labelsize": base_font,
        "legend.fontsize": base_font - 1,
    })

    # --- Bar plot (mean ± std) ---
    means = []
    stds = []
    colors = []
    for i, pipe in enumerate(order):
        vals = df[df[pipeline_col] == pipe][metric_col]
        means.append(vals.mean())
        stds.append(vals.std())
        colors.append(palette[i % len(palette)])

    x_pos = np.arange(len(order))
    bars = ax.bar(
        x_pos, means, yerr=stds, capsize=5,
        color=colors, edgecolor="black", linewidth=0.8,
        alpha=0.85, width=0.65,
        error_kw={"linewidth": 1.5},
    )

    # --- Swarm overlay (black dots) ---
    rng = np.random.RandomState(42)
    for i, pipe in enumerate(order):
        vals = df[df[pipeline_col] == pipe][metric_col].values
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            x_pos[i] + jitter, vals,
            color="black", s=25, zorder=5, alpha=0.7,
            edgecolors="white", linewidths=0.3,
        )

    # --- DummyClassifier baseline ---
    if dummy_score is not None:
        ax.axhline(
            y=dummy_score, color="#E74C3C", linestyle="--",
            linewidth=2, label=f"DummyClassifier = {dummy_score:.3f}",
        )

    # --- Labels & title ---
    ax.set_xticks(x_pos)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=base_font)
    ax.set_ylabel(metric_label, fontsize=base_font + 2, fontweight="bold")
    ax.set_xlabel("")

    title_parts = []
    if model_name:
        title_parts.append(model_name)
    if cohort_name:
        title_parts.append(cohort_name.replace("_", " "))
    if n_test > 0:
        title_parts.append(f"N_test = {n_test}")
    ax.set_title(" | ".join(title_parts), fontsize=base_font + 6, fontweight="bold")

    # --- Grid ---
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # --- Statistical brackets (adjacent pairs only, Holm on ALL pairs) ---
    pairwise = compute_pairwise_wilcoxon(df, pipeline_col, metric_col, split_col)

    if not pairwise.empty:
        y_max = max(m + s for m, s in zip(means, stds))
        bracket_height = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.04
        y_start = y_max + bracket_height * 1.5

        for idx in range(len(order) - 1):
            pipe_a = order[idx]
            pipe_b = order[idx + 1]

            # Find this pair in the pairwise table (order-agnostic)
            row = pairwise[
                ((pairwise["pipeline_a"] == pipe_a) & (pairwise["pipeline_b"] == pipe_b))
                | ((pairwise["pipeline_a"] == pipe_b) & (pairwise["pipeline_b"] == pipe_a))
            ]

            if row.empty:
                continue

            stars = row.iloc[0]["stars"]
            p_adj = row.iloc[0]["p_corrected"]
            y_bracket = y_start + idx * bracket_height * 2.5

            # Format label: stars + p_adj value
            if p_adj < 0.001:
                bracket_label = f"{stars}\np={p_adj:.1e}"
            else:
                bracket_label = f"{stars}\np={p_adj:.3f}"

            # Draw bracket
            ax.plot(
                [x_pos[idx], x_pos[idx], x_pos[idx + 1], x_pos[idx + 1]],
                [y_bracket, y_bracket + bracket_height, y_bracket + bracket_height, y_bracket],
                color="black", linewidth=1.2,
            )
            ax.text(
                (x_pos[idx] + x_pos[idx + 1]) / 2,
                y_bracket + bracket_height * 1.1,
                bracket_label, ha="center", va="bottom",
                fontsize=base_font - 2, fontweight="bold",
            )

        # Adjust y-axis to fit brackets
        ax.set_ylim(top=y_start + len(order) * bracket_height * 3.0 + bracket_height * 3)

    # --- Legend ---
    handles = [
        mpatches.Patch(
            facecolor=colors[i], edgecolor="black",
            label=f"{order[i]}  (μ={means[i]:.3f}±{stds[i]:.3f})"
        )
        for i in range(len(order))
    ]
    if dummy_score is not None:
        from matplotlib.lines import Line2D
        handles.append(
            Line2D([0], [0], color="#E74C3C", linestyle="--", linewidth=2,
                   label=f"DummyClassifier = {dummy_score:.3f}")
        )
    ax.legend(
        handles=handles, loc="upper left",
        framealpha=0.9, fontsize=base_font - 1,
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
        logger.info("Saved figure to {}", save_path)

    return fig


def plot_all_results(
    results: pd.DataFrame,
    metric_col: str = "f1_weighted",
    metric_label: str = "F1 weighted",
    pipeline_col: str = "pipeline",
    split_col: str = "split",
    save_dir: str | Path | None = None,
) -> list[plt.Figure]:
    """Generate one plot per (model, cohort) from full results table.

    Parameters
    ----------
    results : pd.DataFrame
        Full results with columns: model, cohort, pipeline, split, metric_col,
        and optionally dummy_{metric_col}.
    metric_col : str
        Metric column to plot.
    metric_label : str
        Y-axis label.
    pipeline_col : str
        Column with pipeline names.
    split_col : str
        Column with fold/seed identifiers.
    save_dir : str or Path or None
        Directory to save figures. Creates subfolders if needed.

    Returns
    -------
    list[plt.Figure]
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    figs = []

    for (model, cohort), grp in results.groupby(["model", "cohort"]):
        n_test = int(grp["n_test"].iloc[0]) if "n_test" in grp.columns else 0

        # Get DummyClassifier score if present
        dummy_col = f"dummy_{metric_col}"
        dummy_score = float(grp[dummy_col].iloc[0]) if dummy_col in grp.columns else None

        save_path = None
        if save_dir is not None:
            fname = f"{model}_{cohort}_{metric_col}.png"
            save_path = save_dir / fname

        fig = plot_pipeline_comparison(
            df=grp,
            metric_col=metric_col,
            metric_label=metric_label,
            pipeline_col=pipeline_col,
            split_col=split_col,
            model_name=model,
            cohort_name=cohort,
            n_test=n_test,
            dummy_score=dummy_score,
            save_path=save_path,
        )
        figs.append(fig)

    return figs
