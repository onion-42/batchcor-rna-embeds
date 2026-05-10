"""Run model inference: extract Geneformer cell embeddings from tokenized data."""

from __future__ import annotations

import os
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from geneformer import EmbExtractor
from loguru import logger

from batchcor_rna_emb.config import (
    BATCH_SIZE,
    EMB_LABEL_COLS,
    EMBS_LAYER,
    GENEFORMER_EMBEDDINGS_DIR,
    GENEFORMER_TOKENIZED_DIR,
    MAX_CELLS,
    MODEL_PATH,
    NPROC,
)


def _find_dataset_folder(token_dir: Path) -> Path:
    """Locate the HuggingFace dataset folder produced by tokenize_data().

    Parameters
    ----------
    token_dir : Path
        Directory that was passed as output_directory to tokenize_data().

    Returns
    -------
    Path
        Path to the dataset folder containing dataset_info.json.

    Raises
    ------
    AssertionError
        If no valid dataset folder is found.
    """
    folders = [
        p for p in token_dir.iterdir()
        if p.is_dir() and (p / "dataset_info.json").exists()
    ]
    assert folders, f"No HuggingFace dataset found in {token_dir}"
    return folders[0]


def extract_embeddings(
    model_path: str | Path = MODEL_PATH,
    token_dir:  str | Path = GENEFORMER_TOKENIZED_DIR,
    output_dir: str | Path = GENEFORMER_EMBEDDINGS_DIR,
    emb_label_cols: list[str] = EMB_LABEL_COLS,
    embs_layer:  int = EMBS_LAYER,
    batch_size:  int = BATCH_SIZE,
    max_cells:   int | None = MAX_CELLS,
    nproc:       int = NPROC,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract per-cell embeddings from a tokenized dataset using Geneformer.

    Parameters
    ----------
    model_path : Path
        Path to the Geneformer model directory (e.g. Geneformer-V2-104M_CLcancer).
    token_dir : Path
        Directory containing the tokenized HuggingFace dataset.
    output_dir : Path
        Where to save embeddings.npy and metadata.csv.
    emb_label_cols : list[str]
        Obs column names (after tokenizer renaming) to carry through as metadata.
    embs_layer : int
        Which transformer layer to extract (-1 = last hidden layer).
    batch_size : int
        Forward-pass batch size (reduce if OOM).
    max_cells : int or None
        If set, only embed the first N cells (for quick tests).
    nproc : int
        Parallel workers for EmbExtractor.

    Returns
    -------
    embeddings : np.ndarray
        Shape (n_cells, hidden_dim) — float32 embedding matrix.
    metadata : pd.DataFrame
        Shape (n_cells, len(emb_label_cols)) — metadata columns.
    """
    model_path = Path(model_path)
    token_dir = Path(token_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: {}", device)

    # Locate tokenized dataset
    dataset_path = _find_dataset_folder(token_dir)
    logger.info("Loading dataset from: {}", dataset_path)

    dataset = load_from_disk(str(dataset_path))
    logger.info("Dataset columns: {} | Cells: {}",
                dataset.column_names, len(dataset))

    if max_cells:
        dataset = dataset.select(range(max_cells))
        logger.info("Limited to {} cells for testing", max_cells)

    # Build EmbExtractor
    emb_extractor = EmbExtractor(
        model_type="CellClassifier",
        num_classes=0,
        emb_mode="cell",
        cell_emb_style="mean_pool",
        filter_data=None,
        max_ncells=max_cells,
        emb_layer=embs_layer,
        emb_label=emb_label_cols,
        labels_to_plot=[],
        forward_batch_size=batch_size,
        nproc=nproc,
    )

    logger.info("Extracting embeddings from model: {}", model_path.name)
    embs_df = emb_extractor.extract_embs(
        model_directory=str(model_path),
        input_data_file=str(dataset_path),
        output_directory=str(output_dir),
        output_prefix="geneformer_v2",
    )

    logger.info("Raw embs_df shape: {}", embs_df.shape)

    # Separate embedding vectors from metadata columns
    present_label_cols = [c for c in emb_label_cols if c in embs_df.columns]
    emb_cols = [c for c in embs_df.columns if c not in present_label_cols]

    embeddings = embs_df[emb_cols].values.astype(
        np.float32)   # (n_cells, hidden_dim)
    metadata = embs_df[present_label_cols]

    logger.info("Embedding matrix: {} | Metadata columns: {}",
                embeddings.shape, metadata.columns.tolist())

    # Save outputs
    emb_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "metadata.csv"
    np.save(emb_path, embeddings)
    metadata.to_csv(meta_path)
    logger.info("Saved embeddings → {}", emb_path)
    logger.info("Saved metadata   → {}", meta_path)

    return embeddings, metadata


def build_embedding_adata(
    embeddings: np.ndarray,
    metadata:   pd.DataFrame,
    adata_mapped: anndata.AnnData | None = None,
    extra_cols: list[str] | None = None,
) -> anndata.AnnData:
    """Wrap embeddings and metadata into an AnnData object for downstream analysis.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_cells, hidden_dim).
    metadata : pd.DataFrame
        Shape (n_cells, n_label_cols). Must be positionally aligned with embeddings.
    adata_mapped : AnnData, optional
        Source AnnData to pull additional clinical columns from (positional alignment).
    extra_cols : list[str], optional
        Additional columns to copy from adata_mapped.obs (e.g. ['OS', 'PFS', 'Response']).

    Returns
    -------
    anndata.AnnData
        AnnData with embeddings as X and obs populated with metadata.
    """
    adata_emb = anndata.AnnData(X=embeddings)

    # Attach metadata by position (tokenizer preserves order)
    for col in metadata.columns:
        adata_emb.obs[col] = metadata[col].values

    # Attach extra clinical columns from source adata if lengths match
    if adata_mapped is not None and extra_cols:
        if len(adata_emb) == len(adata_mapped):
            for col in extra_cols:
                if col in adata_mapped.obs.columns:
                    adata_emb.obs[col] = adata_mapped.obs[col].values
                    logger.info(
                        "Attached clinical column '{}' from adata_mapped", col)
        else:
            logger.warning(
                "adata_emb ({}) and adata_mapped ({}) have different lengths — "
                "skipping clinical column transfer.",
                len(adata_emb), len(adata_mapped),
            )

    logger.info("Built embedding AnnData: {}", adata_emb.shape)
    return adata_emb
