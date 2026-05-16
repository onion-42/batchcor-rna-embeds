"""
Canonical train / test cohort registry.

Rules (match ``scgpt_embeddings.py`` and curator split for v5):
  * **train** — in-house clinical cohorts used for cAE fitting and CV
    (no ``PUB_`` prefix in raw folder names).
  * **test**  — external / publication cohorts (``PUB_*`` prefix), never used
    to fit cAE or inner CV; only held-out evaluation.

Do not put a cohort in both lists.  ``PUB_KIRC_ICI_combined`` is the same
cohort as ``KIRC_Tissue_ICI_Pred`` (n=1172, identical patient indices) and
must **not** appear in unified train+test objects or OOD metrics.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"

# Raw zarr folder names (without path)
TRAIN_COHORT_NAMES: tuple[str, ...] = (
    "KIRC_Tissue_ICI_Pred.adata",
    "Melanoma_Tissue_ICI_Pred.adata",
    "NSCLC_Tissue_ICI_Pred.adata",
)

TEST_COHORT_NAMES: tuple[str, ...] = (
    "PUB_BLCA_Mariathasan_EGAS00001002556_ICI.adata",
    "PUB_ccRCC_Immotion150_and_151_ICI.adata",
    "PUB_ccRCC_Immotion150_and_151_TKI.adata",
    "PUB_BRCA_SCANB.adata",
)

# Same patients as KIRC_Tissue_ICI_Pred — excluded from unified / OOD (not true OOD).
TEST_COHORT_DUPLICATE_OF_TRAIN: frozenset[str] = frozenset({
    "PUB_KIRC_ICI_combined",
})


def zarr_path(cohort_name: str) -> Path:
    return DATA_RAW / f"{cohort_name}.zarr"


def split_for_cohort(cohort_name: str) -> str:
    """Return ``'train'`` or ``'test'`` for a cohort stem (with or without .zarr)."""
    stem = cohort_name.removesuffix(".zarr").removesuffix(".adata")
    if stem in {n.removesuffix(".adata") for n in TRAIN_COHORT_NAMES}:
        return "train"
    if stem in {n.removesuffix(".adata") for n in TEST_COHORT_NAMES}:
        return "test"
    # Fallback: PUB_ prefix => test (curator convention)
    if stem.startswith("PUB_"):
        return "test"
    return "train"
