"""
scGPT Bulk RNA-seq Embedding Pipeline — Strict Production Version
=================================================================
Rules enforced with zero exceptions:
  1. NO placeholders, NO dummy arrays, NO np.random fallbacks.
  2. NO try/except blocks that swallow errors silently.
  3. Model weights must be manually placed in pretrained/scGPT_human/
  4. Any failure at any stage raises immediately and crashes loudly.

Manual Setup
------------
  1. Download the three model files from scGPT GitHub:
     - best_model.pt
     - vocab.json
     - args.json
  2. Place them in: pretrained/scGPT_human/
  3. Run this script.

Install
-------
    pip install "scgpt>=0.2.1" scanpy anndata zarr torch numpy pandas

Quick-start
-----------
    python scgpt_embeddings.py

    # Override model directory:
    SCGPT_MODEL_DIR=/path/to/weights python scgpt_embeddings.py
"""

# =============================================================================
# STDLIB
# =============================================================================
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# =============================================================================
# THIRD-PARTY  — hard imports; no silent fallbacks
# =============================================================================
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from loguru import logger

# scGPT — if this raises ImportError the script crashes with a clear message.
# That is the correct behaviour; do NOT wrap in try/except.
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab

# =============================================================================
# LOGGING
# =============================================================================
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | {message}"
    ),
    level="INFO",
)

# =============================================================================
# CONFIGURATION
# =============================================================================
REPO_ROOT : Path = Path(__file__).resolve().parents[2]
SCGPT_MODEL_DIR : Path = Path(
    os.environ.get(
        "SCGPT_MODEL_DIR",
        str(REPO_ROOT / "pretrained/scGPT_human")
    )
).resolve()

# Batch inference: 1 is extremely slow (one forward per patient). Use 32 on CPU
# and 64 on CUDA unless overridden via process_all_folders(batch_size=...).
_CUDA_AVAILABLE = torch.cuda.is_available()
BATCH_SIZE: int = 64 if _CUDA_AVAILABLE else 32
N_HVG      : int = 1_200   # hard cap matching scGPT pretraining sequence length
N_BINS     : int = 51      # expression discretisation levels (matches pretraining)
PAD_TOKEN  : str = "<pad>"
CLS_TOKEN  : str = "<cls>"
EOS_TOKEN  : str = "<eos>"
PAD_VALUE  : int = 0       # bin-id assigned to padding positions

EMBEDDINGS_DIR = REPO_ROOT / "embeddings"
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# COHORT PATHS  — edit to match your filesystem layout
# =============================================================================
TRAIN_FOLDERS = [
    REPO_ROOT / "data/raw/KIRC_Tissue_ICI_Pred.adata.zarr",
    REPO_ROOT / "data/raw/Melanoma_Tissue_ICI_Pred.adata.zarr",
    REPO_ROOT / "data/raw/NSCLC_Tissue_ICI_Pred.adata.zarr",
]
TEST_FOLDERS = [
    REPO_ROOT / "data/raw/PUB_BLCA_Mariathasan_EGAS00001002556_ICI.adata.zarr",
    REPO_ROOT / "data/raw/PUB_ccRCC_Immotion150_and_151_ICI.adata.zarr",
    REPO_ROOT / "data/raw/PUB_ccRCC_Immotion150_and_151_TKI.adata.zarr",
]
ALL_FOLDERS = TRAIN_FOLDERS + TEST_FOLDERS


# =============================================================================
# STEP 1 — MODEL ENGINE  (instantiated ONCE, reused across all 6 cohorts)
# =============================================================================

class ScGPTInferenceEngine:
    """
    Owns the GeneVocab and TransformerModel.  Instantiated once in
    process_all_folders() and passed by reference to every cohort call.

    There is NO try/except block inside __init__.  Any failure during
    vocab loading, config parsing, or weight loading raises immediately.
    """

    def __init__(self, model_dir: Path, device: torch.device) -> None:
        self.device    = device
        self.model_dir = model_dir

        logger.info(f"Initialising scGPT engine | dir={model_dir} | device={device}")

        # ── Vocabulary ────────────────────────────────────────────────────────
        vocab_path = model_dir / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"vocab.json not found in {model_dir}. "
                "Please ensure the model files are in the directory."
            )
        self.vocab: GeneVocab = GeneVocab.from_file(str(vocab_path))

        # Ensure special tokens exist (some checkpoints omit them; crash is worse
        # than a loud append + warning).
        for tok in (PAD_TOKEN, CLS_TOKEN, EOS_TOKEN):
            if tok not in self.vocab:
                logger.warning(
                    f"Special token {tok!r} missing from vocab — appending. "
                    "Verify checkpoint compatibility."
                )
                self.vocab.append_token(tok)

        self.pad_id: int = self.vocab[PAD_TOKEN]
        self.cls_id: int = self.vocab[CLS_TOKEN]

        # Lookup: HGNC gene symbol → integer token id
        self.gene2id: Dict[str, int] = {
            g: self.vocab[g]
            for g in self.vocab.get_stoi()
            if not g.startswith("<")
        }
        logger.info(f"Vocabulary loaded: {len(self.gene2id):,} gene tokens.")

        # ── Model hyper-parameters ────────────────────────────────────────────
        config_path = model_dir / "args.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"args.json not found in {model_dir}. "
                "The checkpoint download may be incomplete."
            )
        with open(config_path) as fh:
            cfg = json.load(fh)

        self.embsize: int = cfg.get("embsize", 512)
        logger.info(f"Embedding dimension (d_model): {self.embsize}")

        # ── Build TransformerModel ────────────────────────────────────────────
        self.model = TransformerModel(
            ntoken                = len(self.vocab),
            d_model               = self.embsize,
            nhead                 = cfg.get("nheads", 8),
            d_hid                 = cfg.get("d_hid", 512),
            nlayers               = cfg.get("nlayers", 12),
            vocab                 = self.vocab,
            dropout               = 0.0,            # disabled at inference time
            pad_token             = PAD_TOKEN,
            pad_value             = PAD_VALUE,
            do_mvc                = False,
            do_dab                = False,
            use_batch_labels      = False,
            domain_spec_batchnorm = False,
            explicit_zero_prob    = False,
            use_fast_transformer  = False,  # Disabled for Windows compatibility
            pre_norm              = cfg.get("pre_norm", False),
        )

        # ── Load weights ──────────────────────────────────────────────────────
        weight_path = model_dir / "best_model.pt"
        if not weight_path.exists():
            raise FileNotFoundError(
                f"best_model.pt not found in {model_dir}. "
                "The checkpoint download may be incomplete."
            )
        state = torch.load(str(weight_path), map_location="cpu")
        # Some checkpoint files wrap weights under a "model" sub-key
        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(
                f"{len(missing)} missing weight keys (first 5): {missing[:5]}"
            )
        if unexpected:
            logger.warning(
                f"{len(unexpected)} unexpected weight keys (first 5): {unexpected[:5]}"
            )

        self.model.to(self.device)
        self.model.eval()
        logger.info("scGPT TransformerModel loaded and set to eval mode.")

    @property
    def vocab_gene_set(self) -> frozenset:
        """All HGNC gene symbols known to the model."""
        return frozenset(self.gene2id.keys())


# =============================================================================
# STEP 2 — GENE VOCABULARY MATCHING + HVG SELECTION
# =============================================================================

def intersect_with_vocab(
    adata       : sc.AnnData,
    engine      : ScGPTInferenceEngine,
    n_hvg       : int = N_HVG,
    raw_counts  : Optional[np.ndarray] = None,
) -> Tuple[sc.AnnData, np.ndarray]:
    """
    1. Keep only genes present in the scGPT vocabulary.
    2. If >n_hvg genes remain, select HVGs: ``seurat_v3`` on raw counts when
       ``raw_counts`` is provided (correct for counts); otherwise ``seurat``
       on the (log-normalised) ``adata.X``.
    3. Return filtered AnnData and token-id array aligned with ``adata.var_names``.
    """
    overlap = [g for g in adata.var_names if g in engine.vocab_gene_set]
    logger.info(
        f"Vocabulary match: {len(overlap):,} / {adata.n_vars:,} genes "
        f"({100 * len(overlap) / max(adata.n_vars, 1):.1f}%)"
    )

    if len(overlap) == 0:
        raise ValueError(
            "Zero genes in the dataset match the scGPT vocabulary.\n"
            "scGPT requires HGNC gene symbols (e.g. TP53, CD274, EGFR).\n"
            f"Your adata.var_names look like: {list(adata.var_names[:8])}\n"
            "Fix: adata.var_names = adata.var['gene_name']  (or equivalent column)"
        )

    n_vars_full = adata.n_vars
    col_ix = [adata.var_names.get_loc(g) for g in overlap]

    if raw_counts is not None:
        if raw_counts.shape[0] != adata.n_obs:
            raise ValueError(
                f"raw_counts rows {raw_counts.shape[0]} != n_obs {adata.n_obs}"
            )
        if raw_counts.shape[1] != n_vars_full:
            raise ValueError(
                f"raw_counts cols {raw_counts.shape[1]} != adata n_vars {n_vars_full}"
            )

    adata = adata[:, overlap].copy()

    if adata.n_vars > n_hvg:
        if raw_counts is not None:
            import scipy.sparse as sp

            raw_overlap = raw_counts[:, col_ix]
            if sp.issparse(raw_overlap):
                raw_overlap = raw_overlap.toarray()
            hvg_adata = sc.AnnData(
                X=raw_overlap.astype(np.float32),
                var=adata.var.copy(),
                obs=adata.obs.copy(),
            )
            logger.info(
                f"Selecting top {n_hvg} HVGs with seurat_v3 on raw counts "
                f"from {adata.n_vars} vocab-matched genes …"
            )
            sc.pp.highly_variable_genes(
                hvg_adata, n_top_genes=n_hvg, flavor="seurat_v3"
            )
            hvg_mask = hvg_adata.var["highly_variable"].values
        else:
            logger.warning(
                "No raw_counts for HVG selection — using flavor='seurat' on "
                "log-normalised X (seurat_v3 needs counts)."
            )
            sc.pp.highly_variable_genes(
                adata, n_top_genes=n_hvg, flavor="seurat"
            )
            hvg_mask = adata.var["highly_variable"].values

        adata = adata[:, hvg_mask].copy()
        logger.info(f"HVG selection complete: {adata.n_vars} genes retained.")

    gene_ids = np.array(
        [engine.vocab[g] for g in adata.var_names], dtype=np.int64
    )
    return adata, gene_ids


# =============================================================================
# STEP 3 — PREPROCESSING
# =============================================================================

def preprocess_adata(adata: sc.AnnData) -> Tuple[sc.AnnData, Optional[np.ndarray]]:
    """
    Detect raw vs log-normalised data; apply ``normalize_total`` + ``log1p`` if raw.

    Returns
    -------
    (adata, raw_counts)
        ``raw_counts`` is a dense copy of ``.X`` **before** normalisation when raw
        data was detected (for ``seurat_v3`` HVG selection). Otherwise ``None``.
    """
    import scipy.sparse as sp

    X = adata.X
    sample = X[:50].toarray() if sp.issparse(X) else np.asarray(X[:50])
    sample_max = float(sample.max())

    if sample_max > 14.0:
        logger.info(
            f"max(X[:50]) = {sample_max:.2f} → raw data detected. "
            "Saving raw counts, then normalize_total(1e4) + log1p."
        )
        raw_counts = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float32)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return adata, raw_counts

    logger.info(
        f"max(X[:50]) = {sample_max:.4f} → data appears log-normalised already. "
        "Skipping normalisation (no raw counts for seurat_v3 HVG)."
    )
    return adata, None


# =============================================================================
# STEP 4 — EXPRESSION BINNING (TOKENISATION)
# =============================================================================

def bin_expression(expr: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """
    Discretise a (n_cells, n_genes) float32 matrix into bin indices [0, n_bins-1].

    Binning scheme (identical to scGPT pretraining)
    ------------------------------------------------
    • expr == 0   →  bin 0  (structural zero; always reserved)
    • expr  > 0   →  bins 1 … n_bins-1  via per-cell linear interpolation
                     between min_nonzero and max_nonzero values

    Returns
    -------
    np.ndarray of shape (n_cells, n_genes), dtype int64
    """
    binned = np.zeros_like(expr, dtype=np.int64)

    for i in range(expr.shape[0]):
        row  = expr[i]
        mask = row > 0
        if mask.sum() == 0:
            continue                           # all-zero patient stays all-zero

        vals    = row[mask]
        lo, hi  = vals.min(), vals.max()

        if hi == lo:                           # single unique non-zero value
            binned[i, mask] = 1
            continue

        idx = np.floor(
            (vals - lo) / (hi - lo) * (n_bins - 2)
        ).astype(np.int64) + 1                 # shift by 1 so 0 stays reserved

        binned[i, mask] = np.clip(idx, 1, n_bins - 1)

    return binned   # (n_cells, n_genes)


def build_model_inputs(
    expr_binned : np.ndarray,   # (n_cells, n_genes)  int64
    gene_ids    : np.ndarray,   # (n_genes,)           int64
    pad_id      : int,
    n_bins      : int = N_BINS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct the three tensors consumed by TransformerModel.forward().

    Tensor layout  (seq_len = n_genes + 1)
    ----------------------------------------
    position 0       → CLS token
    positions 1…L    → real gene tokens (identical across all cells in batch)

    key_padding_mask uses PyTorch convention: True = ignore (padding).
    All positions are False here because every patient has the same HVG set
    (no variable-length padding is needed between patients).
    """
    n_cells, n_genes = expr_binned.shape
    seq_len          = n_genes + 1

    # Integer bin ids must be long tensors for scGPT's value encoder (nn.Embedding).
    gene_tokens  = torch.full((n_cells, seq_len), pad_id, dtype=torch.long)
    expr_values  = torch.full((n_cells, seq_len), PAD_VALUE, dtype=torch.long)
    key_pad_mask = torch.zeros((n_cells, seq_len), dtype=torch.bool)

    # ``torch.tensor`` avoids environments where ``torch.from_numpy`` fails
    # with "Numpy is not available" on some Windows / minimal torch builds.
    g_tensor = torch.tensor(np.asarray(gene_ids), dtype=torch.long)
    b_tensor = torch.tensor(np.asarray(expr_binned), dtype=torch.long)

    # Position 0 — CLS
    # The model uses pad_id as the input token for the CLS position and
    # replaces it with its own learned CLS embedding internally.
    gene_tokens [:, 0] = pad_id
    expr_values [:, 0] = n_bins   # out-of-range sentinel the model recognises

    # Positions 1…L — gene tokens
    gene_tokens [:, 1:] = g_tensor.unsqueeze(0).expand(n_cells, -1)
    expr_values [:, 1:] = b_tensor

    return gene_tokens, expr_values, key_pad_mask


# =============================================================================
# STEP 5 — BATCHED GPU INFERENCE + POOLING
# =============================================================================

def _pool(
    h            : torch.Tensor,  # (B, L, d_model)
    key_pad_mask : torch.Tensor,  # (B, L)  True = padding position
    strategy     : str,
) -> torch.Tensor:                # (B, d_model)
    """Collapse the gene-token sequence dimension into a single patient vector."""
    if strategy == "cls":
        # CLS is always at position 0; this is the canonical scGPT embedding.
        return h[:, 0, :]

    if strategy == "mean":
        real   = (~key_pad_mask).unsqueeze(-1).float()  # (B, L, 1)
        summed = (h * real).sum(dim=1)                  # (B, d_model)
        counts = real.sum(dim=1).clamp(min=1e-9)        # (B, 1)
        return summed / counts

    raise ValueError(
        f"pooling='{strategy}' is not supported. Choose 'cls' or 'mean'."
    )


def extract_embeddings_batched(
    gene_tokens  : torch.Tensor,
    expr_values  : torch.Tensor,
    key_pad_mask : torch.Tensor,
    engine       : ScGPTInferenceEngine,
    batch_size   : int = BATCH_SIZE,
    pooling      : str = "cls",
) -> np.ndarray:
    """
    Run batched forward passes through the scGPT TransformerModel.

    OOM behaviour
    -------------
    torch.cuda.OutOfMemoryError is NOT caught here.  If it occurs, re-run
    with a smaller batch_size (e.g. 8 or 4).

    Returns
    -------
    np.ndarray of shape (n_samples, engine.embsize), dtype float32
    """
    n_samples  = gene_tokens.shape[0]
    embeddings = np.empty((n_samples, engine.embsize), dtype=np.float32)
    device     = engine.device

    logger.info(
        f"Starting forward pass | patients={n_samples} | "
        f"batch_size={batch_size} | pooling='{pooling}' | device={device}"
    )

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)

            g_tok  = gene_tokens [start:end].to(device)
            e_val  = expr_values [start:end].to(device)
            k_mask = key_pad_mask[start:end].to(device)

            # Forward pass — NO try/except; errors surface immediately
            output = engine.model(
                src                  = g_tok,
                values               = e_val,
                src_key_padding_mask = k_mask,
                CLS                  = True,
                CCE                  = False,
                MVC                  = False,
                ECS                  = False,
                do_sample            = False,
            )

            # ── Resolve output format across scGPT versions ───────────────────
            # Newer scGPT (≥0.2) returns a dict with "cell_emb" already pooled.
            # Older versions return the raw hidden-state tensor or a dataclass.
            if isinstance(output, dict):
                if "cell_emb" in output:
                    # Already pooled to (B, d_model)
                    batch_emb = output["cell_emb"].float()
                elif "h" in output:
                    batch_emb = _pool(output["h"].float(), k_mask, pooling)
                else:
                    raise RuntimeError(
                        f"Unrecognised output keys from scGPT: {list(output.keys())}.\n"
                        "Expected 'cell_emb' or 'h'.  Check your scgpt version:\n"
                        "  python -c \"import scgpt; print(scgpt.__version__)\""
                    )
            elif hasattr(output, "last_hidden_state"):
                # HuggingFace-style ModelOutput
                batch_emb = _pool(
                    output.last_hidden_state.float(), k_mask, pooling
                )
            elif isinstance(output, torch.Tensor):
                if output.dim() == 3:   # (B, L, d_model)
                    batch_emb = _pool(output.float(), k_mask, pooling)
                elif output.dim() == 2: # (B, d_model) already pooled
                    batch_emb = output.float()
                else:
                    raise RuntimeError(
                        f"Unexpected tensor shape from scGPT model: {output.shape}"
                    )
            else:
                raise RuntimeError(
                    f"Cannot parse scGPT model output of type {type(output)}.\n"
                    "Please check your scgpt library version."
                )

            embeddings[start:end] = batch_emb.cpu().numpy()

            if end % (batch_size * 5) == 0 or end == n_samples:
                logger.info(
                    f"  [{end:>{len(str(n_samples))}}/{n_samples}] "
                    f"{100*end/n_samples:.0f}% complete"
                )

    return embeddings


# =============================================================================
# MAIN TRANSFORMATION — raw AnnData → dense patient embeddings
# =============================================================================

def transform_to_scgpt_embeddings(
    adata      : sc.AnnData,
    engine     : ScGPTInferenceEngine,
    batch_size : int = BATCH_SIZE,
    pooling    : str = "cls",
) -> np.ndarray:
    """
    End-to-end pipeline:
        raw AnnData  →  (n_patients, embedding_dim)  float32 numpy array

    Stages
    ------
    1. Preprocessing    — normalize_total + log1p (if data is raw)
    2. Vocab + HVG      — filter to model vocabulary, select top 1200 HVGs
    3. Binning          — continuous values → discrete token ids
    4. Tensor assembly  — CLS + gene tokens + key_padding_mask
    5. Batched inference— GPU forward pass, CLS pooling

    NO fallbacks.  NO random arrays.  Every exception propagates.
    """
    import scipy.sparse as sp

    # 1 — Preprocess (optional raw matrix for correct HVG on counts)
    adata, raw_counts = preprocess_adata(adata)

    # 2 — Vocabulary match + HVG selection
    adata, gene_ids = intersect_with_vocab(
        adata, engine, n_hvg=N_HVG, raw_counts=raw_counts
    )

    # 3 — Densify and bin
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    logger.info(
        f"Binning expression matrix: "
        f"{X.shape[0]:,} patients × {X.shape[1]:,} genes"
    )
    expr_binned = bin_expression(X, n_bins=N_BINS)   # (n_patients, n_genes)

    # 4 — Build model input tensors
    logger.info("Assembling model input tensors …")
    gene_tokens, expr_values, key_pad_mask = build_model_inputs(
        expr_binned, gene_ids, pad_id=engine.pad_id, n_bins=N_BINS
    )

    # 5 — Batched forward pass
    embeddings = extract_embeddings_batched(
        gene_tokens, expr_values, key_pad_mask,
        engine     = engine,
        batch_size = batch_size,
        pooling    = pooling,
    )

    logger.info(
        f"Embeddings ready: shape={embeddings.shape}  dtype={embeddings.dtype}"
    )
    return embeddings


# =============================================================================
# DATA LOADING  — crashes loudly if the path is wrong or the file is corrupt
# =============================================================================

def load_data_from_folder(folder_path: Path) -> sc.AnnData:
    if not folder_path.exists():
        raise FileNotFoundError(
            f"Zarr dataset not found: {folder_path}\n"
            "Check the path and ensure the storage volume is mounted."
        )
    logger.info(f"Loading: {folder_path}")
    try:
        adata = ad.read_zarr(str(folder_path))
    except KeyError as e:
        # Zarr metadata corruption: read with raw zarr and reconstruct
        logger.warning(
            f"Zarr read failed with KeyError: {e}. "
            "Reading with raw zarr and reconstructing..."
        )
        import zarr
        import numpy as np
        root = zarr.open(str(folder_path), mode='r')
        
        # Read X (expression matrix)
        X = np.array(root['X'][:])
        
        # Read var._index (gene names)
        var_index = root['var']['_index'][:] if '_index' in root['var'] else None
        if var_index is None:
            var_index = np.array([f"gene_{i}" for i in range(X.shape[1])])
        else:
            var_index = np.array([name.decode() if isinstance(name, bytes) else name for name in var_index])
        
        # Read obs._index (sample names) - but might not exist
        obs_index = root['obs']['_index'][:] if 'obs' in root and '_index' in root['obs'] else None
        if obs_index is None:
            obs_index = np.array([f"cell_{i}" for i in range(X.shape[0])])
        else:
            obs_index = np.array([name.decode() if isinstance(name, bytes) else name for name in obs_index])
        
        # Create AnnData object
        adata = ad.AnnData(X=X)
        adata.var.index = var_index
        adata.obs.index = obs_index
        
        logger.info(f"  Reconstructed: {X.shape[0]:,} cells × {X.shape[1]:,} genes (from var._index)")
    
    logger.info(f"  → {adata.n_obs:,} patients × {adata.n_vars:,} genes")
    return adata


# =============================================================================
# PER-COHORT PROCESSING
# =============================================================================

def process_single_folder(
    folder_path : Path,
    engine      : ScGPTInferenceEngine,
    batch_size  : int = BATCH_SIZE,
    pooling     : str = "cls",
) -> None:
    """Process one cohort Zarr dataset and write .npy + patient-ID .csv."""
    # Safe suffix stripping (avoid str.replace corrupting in-folder ".adata" text)
    cohort_name = folder_path.name.removesuffix(".zarr").removesuffix(".adata")

    logger.info(f"{'='*60}")
    logger.info(f"Cohort: {cohort_name}")
    logger.info(f"{'='*60}")

    adata      = load_data_from_folder(folder_path)
    embeddings = transform_to_scgpt_embeddings(
        adata, engine, batch_size=batch_size, pooling=pooling
    )

    # ── Save embeddings (.npy) ────────────────────────────────────────────────
    npy_path = EMBEDDINGS_DIR / f"{cohort_name}_scgpt_embeddings.npy"
    np.save(str(npy_path), embeddings)
    logger.info(f"Saved embeddings  → {npy_path}  shape={embeddings.shape}")

    # ── Save patient-ID index (.csv) ─────────────────────────────────────────
    # Keeps embeddings traceable to the original obs_names (patient barcodes).
    csv_path = EMBEDDINGS_DIR / f"{cohort_name}_patient_ids.csv"
    pd.DataFrame({
        "patient_id" : adata.obs_names.tolist(),
        "cohort"     : cohort_name,
        "emb_row_idx": np.arange(adata.n_obs),
    }).to_csv(csv_path, index=False)
    logger.info(f"Saved patient IDs → {csv_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def process_all_folders(
    batch_size : int = BATCH_SIZE,
    pooling    : str = "cls",
) -> None:
    """
    Orchestrator:
      1. Verify model weights exist locally.
      2. Load GPU + model once.
      3. Process all 6 cohorts sequentially.
      4. Crash immediately on any error — no per-cohort recovery.
    """
    # ── Step 0: verify model weights are in place ────────────────────────────
    weight_path = SCGPT_MODEL_DIR / "best_model.pt"
    if not weight_path.exists():
        print(
            f"ERROR: Model weight file not found.\n"
            f"Expected location: {weight_path.resolve()}\n"
            f"Please ensure the model files are in the directory:\n"
            f"  - best_model.pt\n"
            f"  - vocab.json\n"
            f"  - args.json"
        )
        sys.exit(1)
    logger.info(f"Model weights verified: {weight_path}")

    # ── Detect compute device ─────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning(
            "No CUDA GPU detected — running on CPU.\n"
            "This will be slow.  Reduce batch_size to 4 if you run out of RAM."
        )

    # ── Load model ONCE for all cohorts ──────────────────────────────────────
    engine = ScGPTInferenceEngine(model_dir=SCGPT_MODEL_DIR, device=device)

    # ── Iterate cohorts — no try/except; first failure stops everything ───────
    for i, folder in enumerate(ALL_FOLDERS, start=1):
        logger.info(f"Cohort {i}/{len(ALL_FOLDERS)}: {folder.name}")
        process_single_folder(
            folder, engine,
            batch_size = batch_size,
            pooling    = pooling,
        )

    logger.info(
        f"\nAll {len(ALL_FOLDERS)} cohorts complete.\n"
        f"Embeddings saved to: {EMBEDDINGS_DIR.resolve()}"
    )


if __name__ == "__main__":
    process_all_folders(
        batch_size = BATCH_SIZE,
        pooling    = "cls",    # swap to "mean" for gene-level mean pooling
    )