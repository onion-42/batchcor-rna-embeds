"""Константы, пути и ключи проекта batch correction RNA embeddings."""
from __future__ import annotations

from pathlib import Path

# --- Reproducibility ---
SEED: int = 42

# --- Paths ---
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_INTERIM_DIR: Path = DATA_DIR / "interim"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# --- AnnData keys ---
BATCH_COL: str = "batch"
DIAGNOSIS_COL: str = "diagnosis"
SPLIT_PREFIX: str = "Split_"
TARGET_PREFIX: str = "Target_"

# Embedding keys in .obsm (FM = Foundation Model)
EMBEDDING_KEY_PATTERN: str = "FM_{model}_embeddings"

# Keys written by correction modules into .obsm
HARMONY_SUFFIX: str = "_Harmony"
DANN_SUFFIX: str = "_DANN"

# --- Stress test ---
STRESS_LEVELS: list[str] = ["sanity", "weak_ood", "true_ood"]

# --- Modeling defaults ---
N_PCA_COMPONENTS: int = 100
CV_N_SPLITS: int = 5
LAMA_TIMEOUT: int = 300

# --- Metric aggregation ---
PROBA_PREFIX: str = "Proba_"
