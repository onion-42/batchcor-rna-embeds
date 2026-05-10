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
DATA_INTERIM_PT_DIR: Path = DATA_DIR / "interim_PT"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# --- Paths (Colab / Google Drive) ---
COLAB_ROOT_DIR = Path("/content")
DRIVE_DIR = Path("/content/drive/MyDrive/BG_Internship_group_7")

GENEFORMER_INPUT_DIR = COLAB_ROOT_DIR / "data" / "geneformer_input"
GENEFORMER_TOKENIZED_DIR = COLAB_ROOT_DIR / "data" / "geneformer_tokenized"
GENEFORMER_EMBEDDINGS_DIR = COLAB_ROOT_DIR / "data" / "geneformer_embeddings"

MODEL_DIR = Path("/content/Geneformer")
MODEL_PATH = MODEL_DIR / "Geneformer-V2-104M_CLcancer"

# --- Dataset ---
DATASET_FILENAME = "NSCLC_Tissue_ICI_Pred.adata.zarr"
DATASET_PATH = DATA_RAW_DIR / DATASET_FILENAME

# --- AnnData keys ---
BATCH_COL: str = "RNA_batch"
DIAGNOSIS_COL: str = "Diagnosis"
RESPONSE_COL: str = "Response"
OS_COL: str = "OS"
PFS_COL: str = "PFS"
THERAPY_COL: str = "Therapy"
SPLIT_PREFIX: str = "Split_"
TARGET_PREFIX: str = "Target_"

# Embedding keys in .obsm (FM = Foundation Model)
EMBEDDING_KEY_PATTERN: str = "FM_{model}_embeddings"

# --- COMPASS Foundation Model ---
COMPASS_MODEL_NAME: str = "FM_COMPASS"
COMPASS_EMBEDDING_KEY: str = "FM_COMPASS_embedding"
COMPASS_PCA_KEY: str = "PCA128d_FM_COMPASS_embedding"
COMPASS_UMAP_KEY: str = "UMAP3d_FM_COMPASS_embedding"
COMPASS_METADATA_KEY: str = "FM_COMPASS_metadata"
COMPASS_MODEL_URL: str = (
    "https://www.immuno-compass.com/download/model/finetuner_pft_all.pt"
)
COMPASS_N_PCA: int = 128
COMPASS_N_UMAP: int = 3

# --- COMPASS PreTrainer (no ICI fine-tuning, no data leakage) ---
COMPASS_PT_MODEL_NAME: str = "FM_COMPASS_PT"
COMPASS_PT_EMBEDDING_KEY: str = "FM_COMPASS_PT_embedding"
COMPASS_PT_PCA_KEY: str = "PCA128d_FM_COMPASS_PT_embedding"
COMPASS_PT_UMAP_KEY: str = "UMAP3d_FM_COMPASS_PT_embedding"
COMPASS_PT_METADATA_KEY: str = "FM_COMPASS_PT_metadata"
COMPASS_PT_MODEL_URL: str = (
    "https://www.immuno-compass.com/download/model/pretrainer.pt"
)

# Cancer code mapping for COMPASS (TCGA abbreviations)
COHORT_CANCER_CODE: dict[str, str] = {
    "PUB_KIRC_ICI_combined": "KIRC",
    "Melanoma_Tissue_ICI_Pred": "SKCM",
    "NSCLC_Tissue_ICI_Pred": "LUAD",
    "PUB_BLCA_Mariathasan_EGAS00001002556": "BLCA",
    "PUB_ccRCC_Immotion150_and_151_ICI": "KIRC",
    "PUB_ccRCC_Immotion150_and_151_TKI": "KIRC",
}

# Keys written by correction modules into .obsm
HARMONY_SUFFIX: str = "_Harmony"
DANN_SUFFIX: str = "_DANN"

# --- Geneformer tokenizer ---
TOKENIZER_ATTR_DICT: dict[str, str] = {
    BATCH_COL: "batch",
    DIAGNOSIS_COL: "diagnosis",
}
EMB_LABEL_COLS: list[str] = list(
    TOKENIZER_ATTR_DICT.values())  # ["batch", "diagnosis"]

# --- Geneformer / EmbExtractor settings ---
BATCH_SIZE: int = 16      # lower to 8 if OOM on Colab
EMBS_LAYER: int = -1      # last hidden layer
MAX_CELLS: int | None = None  # set an int to limit cells during quick tests
NPROC: int = 2

# --- UMAP settings ---
UMAP_N_NEIGHBORS: int = 15
UMAP_METRIC: str = "cosine"
UMAP_MIN_DIST: float = 0.3

# --- Imputation ---
ZERO_IMPUTE_VALUE: float = 0.00001

# --- Stress test ---
STRESS_LEVELS: list[str] = ["sanity", "weak_ood", "true_ood"]

# --- Modeling defaults ---
N_PCA_COMPONENTS: int = 100
CV_N_SPLITS: int = 5
LAMA_TIMEOUT: int = 300

# --- Metric aggregation ---
PROBA_PREFIX: str = "Proba_"
