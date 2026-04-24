"""Visualization utilities: UMAP, heatmaps, ROC curves, KM plots, decay plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from loguru import logger
from matplotlib.figure import Figure


def plot_umap_grid(
    adatas: dict[str, sc.AnnData],
    color_cols: list[str],
    basis_key: str = "X_umap",
    figsize: tuple[float, float] = (5, 5),
    point_size: int = 5,
    save_path: str | None = None,
) -> Figure:
    """
    Plot a grid of UMAP embeddings: methods (rows) x color annotations (cols).

    Parameters
    ----------
    adatas : dict[str, sc.AnnData]
        Method name -> AnnData with UMAP coordinates.
    color_cols : list[str]
        Columns in ``.obs`` to color by (e.g. ['batch', 'diagnosis']).
    basis_key : str
        Key in ``.obsm`` for UMAP coordinates.
    figsize : tuple[float, float]
        Figure size per subplot.
    point_size : int
        Point size in scatter plots.
    save_path : str or None
        Path to save the figure.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    n_methods = len(adatas)
    n_colors = len(color_cols)
    fig, axes = plt.subplots(
        n_methods, n_colors,
        figsize=(figsize[0] * n_colors, figsize[1] * n_methods),
        squeeze=False,
    )

    for i, (method_name, adata) in enumerate(adatas.items()):
        for j, col in enumerate(color_cols):
            ax = axes[i, j]
            if basis_key in adata.obsm and col in adata.obs.columns:
                coords = np.asarray(adata.obsm[basis_key])
                categories = adata.obs[col].astype(str)
                unique_cats = sorted(categories.unique())
                cmap = plt.cm.get_cmap("tab20", len(unique_cats))

                for k, cat in enumerate(unique_cats):
                    mask = categories == cat
                    ax.scatter(
                        coords[mask, 0], coords[mask, 1],
                        s=point_size, alpha=0.6, label=cat, c=[cmap(k)],
                    )

                if len(unique_cats) <= 10:
                    ax.legend(fontsize=6, markerscale=2, loc="best")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

            if i == 0:
                ax.set_title(col, fontsize=12)
            if j == 0:
                ax.set_ylabel(method_name, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("UMAP grid saved to '{}'", save_path)
    return fig


def plot_metrics_heatmap(
    df: pd.DataFrame,
    title: str = "Batch Correction Metrics",
    figsize: tuple[float, float] = (10, 6),
    save_path: str | None = None,
) -> Figure:
    """
    Plot an annotated heatmap of metric values.

    Parameters
    ----------
    df : pd.DataFrame
        Metrics table (methods as rows, metrics as columns).
    title : str
        Figure title.
    figsize : tuple[float, float]
        Figure size.
    save_path : str or None
        Path to save figure.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df.select_dtypes(include=[np.number]),
        annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0, vmax=1, ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Heatmap saved to '{}'", save_path)
    return fig


def plot_scatter_avg_batch_bio(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (7, 6),
    save_path: str | None = None,
) -> Figure:
    """
    Scatter plot of AvgBATCH vs AvgBio per method.

    Parameters
    ----------
    df : pd.DataFrame
        Metrics table with 'AvgBATCH' and 'AvgBio' columns.
    figsize : tuple[float, float]
        Figure size.
    save_path : str or None
        Path to save figure.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(df["AvgBATCH"], df["AvgBio"], s=100, zorder=3)
    for method_name, row in df.iterrows():
        ax.annotate(
            str(method_name),
            (row["AvgBATCH"], row["AvgBio"]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=10,
        )

    ax.set_xlabel("AvgBATCH (batch mixing)", fontsize=12)
    ax.set_ylabel("AvgBio (bio preservation)", fontsize=12)
    ax.set_title("Batch Correction Trade-off", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Scatter plot saved to '{}'", save_path)
    return fig


def plot_roc_curves(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "ROC Curves",
    figsize: tuple[float, float] = (7, 6),
    save_path: str | None = None,
) -> Figure:
    """
    Plot ROC curves for multiple models/conditions.

    Parameters
    ----------
    results : dict[str, tuple[np.ndarray, np.ndarray]]
        Label -> (y_true, y_prob) pairs.
    title : str
        Plot title.
    figsize : tuple[float, float]
        Figure size.
    save_path : str or None
        Path to save figure.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=figsize)

    for label, (y_true, y_prob) in results.items():
        n_classes = len(np.unique(y_true))
        if n_classes < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("ROC curves saved to '{}'", save_path)
    return fig


def plot_generalization_decay(
    results_df: pd.DataFrame,
    metric_col: str = "f1_weighted",
    level_col: str = "level",
    group_col: str = "model",
    figsize: tuple[float, float] = (8, 5),
    save_path: str | None = None,
) -> Figure:
    """
    Plot generalization decay: metric vs stress level.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results table with stress level, metric, and grouping columns.
    metric_col : str
        Metric to plot on y-axis.
    level_col : str
        Column for stress levels.
    group_col : str
        Column for grouping (e.g. model name).
    figsize : tuple[float, float]
        Figure size.
    save_path : str or None
        Path to save figure.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    level_order = ["sanity", "weak_ood", "true_ood"]
    level_labels = {"sanity": "Sanity Check", "weak_ood": "Weak OOD", "true_ood": "True OOD"}

    for group_name, group_df in results_df.groupby(group_col):
        values = []
        for level in level_order:
            level_data = group_df[group_df[level_col] == level]
            if not level_data.empty:
                values.append(float(level_data[metric_col].mean()))
            else:
                values.append(np.nan)

        ax.plot(
            [level_labels.get(l, l) for l in level_order],
            values,
            marker="o", linewidth=2, label=str(group_name),
        )

    ax.set_xlabel("Stress Level", fontsize=12)
    ax.set_ylabel(metric_col, fontsize=12)
    ax.set_title("Generalization Decay", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Generalization decay plot saved to '{}'", save_path)
    return fig


def plot_loss_curves(
    history: dict[str, list[float]],
    title: str = "DANN Training Loss",
    figsize: tuple[float, float] = (8, 5),
    save_path: str | None = None,
) -> Figure:
    """
    Plot training loss curves from DANN history.

    Parameters
    ----------
    history : dict[str, list[float]]
        Loss name -> list of per-epoch values.
    title : str
        Plot title.
    figsize : tuple[float, float]
        Figure size.
    save_path : str or None
        Path to save figure.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, values in history.items():
        if values:
            ax.plot(values, label=name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Loss curves saved to '{}'", save_path)
    return fig
