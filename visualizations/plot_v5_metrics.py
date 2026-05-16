"""
visualizations/plot_v5_metrics.py
=================================
Bar chart: in-distribution CV mean ROC-AUC vs per-cohort OOD ROC-AUC (v5 strict).

Inputs (from ``v5_strict_pipeline``):
  metrics_csv/v5_os_bin35_summary.csv
  metrics_csv/v5_os_bin35_ood_per_cohort.csv

Output:
  visualizations/v5_final_ood_performance.png

Usage::

    python visualizations/plot_v5_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "metrics_csv"
SUMMARY_CSV = METRICS_DIR / "v5_os_bin35_summary.csv"
OOD_CSV = METRICS_DIR / "v5_os_bin35_ood_per_cohort.csv"
OUT_PNG = REPO_ROOT / "visualizations" / "v5_final_ood_performance.png"

# Palette
CV_COLOR = "#2E86AB"
OOD_COLOR = "#A23B72"
SKIP_COLOR = "#B8B8B8"
REF_LINE = 0.5


def _short_cohort(name: str) -> str:
    mapping = {
        "PUB_BRCA_SCANB": "BRCA (SCANB)",
        "PUB_KIRC_ICI_combined": "KIRC ICI",
        "PUB_ccRCC_Immotion150_and_151_ICI": "ccRCC ICI",
        "PUB_ccRCC_Immotion150_and_151_TKI": "ccRCC TKI",
        "PUB_BLCA_Mariathasan_EGAS00001002556_ICI": "BLCA ICI",
    }
    return mapping.get(name, name.replace("PUB_", "").replace("_", " ")[:28])


def load_metrics() -> tuple[float, float, pd.DataFrame]:
    if not SUMMARY_CSV.is_file():
        raise FileNotFoundError(f"Missing {SUMMARY_CSV} — run v5_strict_pipeline first.")
    if not OOD_CSV.is_file():
        raise FileNotFoundError(f"Missing {OOD_CSV} — run v5_strict_pipeline first.")

    summary = pd.read_csv(SUMMARY_CSV)
    ood = pd.read_csv(OOD_CSV)
    mean_cv = float(summary["mean_roc_auc"].iloc[0])
    std_cv = float(summary["std_roc_auc"].iloc[0])
    return mean_cv, std_cv, ood


def plot_v5_performance(
    mean_cv: float,
    std_cv: float,
    ood: pd.DataFrame,
    out_path: Path = OUT_PNG,
) -> Path:
    labels = ["Mean CV\n(train ICI)"]
    values: list[float] = [mean_cv]
    colors: list[str] = [CV_COLOR]
    errs: list[float] = [std_cv]

    ood = ood.sort_values("cohort")
    for _, row in ood.iterrows():
        labels.append(_short_cohort(str(row["cohort"])))
        auc = row["roc_auc"]
        if pd.isna(auc):
            values.append(0.0)
            colors.append(SKIP_COLOR)
            errs.append(0.0)
        else:
            values.append(float(auc))
            colors.append(OOD_COLOR)
            errs.append(0.0)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.15), 6.2))

    bars = ax.bar(
        x,
        values,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        width=0.72,
        zorder=3,
    )
    ax.errorbar(
        x[0],
        mean_cv,
        yerr=std_cv,
        fmt="none",
        ecolor="#1a1a1a",
        capsize=8,
        capthick=1.5,
        zorder=4,
    )

    for i, (bar, val) in enumerate(zip(bars, values)):
        if i == 0:
            label = f"{val:.3f} ± {std_cv:.3f}"
        elif colors[i] == SKIP_COLOR:
            label = "N/A"
        else:
            label = f"{val:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="600",
        )

    ax.axhline(REF_LINE, color="#666666", linestyle="--", linewidth=1, alpha=0.7, zorder=1)
    ax.set_ylim(0, min(1.05, max(values + [REF_LINE]) + 0.18))
    ax.set_ylabel("ROC-AUC", fontsize=12, fontweight="600")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=10)
    ax.set_title(
        "v5 strict pipeline — OS at 35 months\n"
        "In-fold CV (no leakage) vs per-cohort OOD",
        fontsize=14,
        fontweight="700",
        pad=16,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.35, linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)

    legend_handles = [
        Patch(facecolor=CV_COLOR, label="5-fold CV (train)"),
        Patch(facecolor=OOD_COLOR, label="OOD per test cohort"),
        Patch(facecolor=SKIP_COLOR, label="Skipped (no labels / single class)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> int:
    mean_cv, std_cv, ood = load_metrics()
    out = plot_v5_performance(mean_cv, std_cv, ood)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
