"""Code to create features for modeling.

Covers:
- HUGO → Ensembl gene ID mapping via MyGene.info
- Zero-value imputation
- Pseudo-count preparation for Geneformer tokenization
- TranscriptomeTokenizer wrapper
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import anndata as ad
import mygene
import numpy as np
import pandas as pd
import scipy.sparse as sp
from loguru import logger

from batchcor_rna_emb.config import (
    NPROC,
    TOKENIZER_ATTR_DICT,
    ZERO_IMPUTE_VALUE,
)


# ── 1. Gene symbol → Ensembl ID mapping ───────────────────────────────────────

def fetch_ensembl_ids(
    genes: Iterable[str],
    species: str = "human",
) -> pd.Series:
    """Fetch Ensembl IDs for HUGO symbols from MyGene.info.

    Parameters
    ----------
    genes : Iterable[str]
        HUGO gene symbols to query.
    species : str
        Species string accepted by MyGene.info (default: "human").

    Returns
    -------
    pd.Series
        Index = HUGO symbol, values = Ensembl gene ID (ENSG…).
        Symbols with no hit are dropped.
    """
    mg = mygene.MyGeneInfo()

    # Sanitize: replace empty/NaN with a dummy so the API doesn't crash
    clean_genes = [
        str(g).strip() if pd.notna(g) and str(
            g).strip() != "" else "DUMMY_GENE"
        for g in genes
    ]

    df = mg.querymany(
        clean_genes,
        species=species,
        scopes=["symbol"],
        fields=["ensembl.gene"],
        as_dataframe=True,
        df_index=True,
        verbose=False,
    )

    if "ensembl" not in df.columns or df.empty:
        return pd.Series(dtype=object)

    # Drop duplicated index entries (multi-match genes)
    df = df[~df.index.duplicated(keep="first")]

    def _extract_ensg(val):
        if isinstance(val, str):
            return val if val.startswith("ENSG") else None
        if isinstance(val, dict):
            return val.get("gene")
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) > 0:
                first = val[0]
                if isinstance(first, dict):
                    return first.get("gene")
                if isinstance(first, str):
                    return first if first.startswith("ENSG") else None
            return None
        return None

    return df["ensembl"].apply(_extract_ensg).dropna()


def rename_adata_vars_to_ensembl(
    adata: ad.AnnData,
    drop_missing: bool = True,
) -> tuple[ad.AnnData, set[str], set[str]]:
    """Rename AnnData var_names from HUGO symbols to Ensembl IDs.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object whose var_names hold HUGO/HGNC gene symbols.
    drop_missing : bool, default=True
        If True, genes with no resolved Ensembl ID are dropped.
        Strongly recommended for Geneformer, which only recognises ENSG IDs.

    Returns
    -------
    renamed_adata : ad.AnnData
        AnnData with updated var_names set to Ensembl IDs.
    found : set[str]
        HUGO symbols successfully resolved.
    missing : set[str]
        HUGO symbols left unresolved.
    """
    hugo_genes = adata.var_names.tolist()
    logger.info(
        "Querying MyGene to map {} HUGO symbols to Ensembl...", len(
            hugo_genes),
    )

    ensembl_series = fetch_ensembl_ids(hugo_genes)
    found = set(ensembl_series.index)
    missing = set(hugo_genes) - found

    logger.info("Resolved: {} | Missing/Unmapped: {}",
                len(found), len(missing))

    # Preserve original HUGO symbols in var metadata
    adata.var["hugo_symbol"] = adata.var_names

    mapper = {gene: ensembl_series.get(gene, gene) for gene in hugo_genes}

    if drop_missing and missing:
        logger.info("Dropping {} unmapped genes...", len(missing))
        keep_mask = [g in found for g in adata.var_names]
        adata = adata[:, keep_mask].copy()

    adata.var_names = [mapper[g] for g in adata.var_names]
    adata.var_names_make_unique()

    return adata, found, missing


# ── 2. Imputation & pseudo-count preparation ───────────────────────────────────

def impute_zeros(
    expressions: pd.DataFrame,
    value: float = ZERO_IMPUTE_VALUE,
) -> pd.DataFrame:
    """Replace zero values with a small positive constant for log-stability.

    Parameters
    ----------
    expressions : pd.DataFrame
        Cells × genes expression matrix (TPM or similar).
    value : float
        Replacement value for zeros (default from config).

    Returns
    -------
    pd.DataFrame
        Imputed DataFrame (copy).
    """
    imputed = expressions.copy()
    n_zeros = int((expressions == 0).sum().sum())
    imputed.replace(0, value, inplace=True)
    logger.info("Imputed {} zero values with {}", n_zeros, value)
    return imputed


def prepare_pseudo_counts(
    adata: ad.AnnData,
    device: str = "cpu",
) -> ad.AnnData:
    """Convert expression matrix to pseudo-integer counts for Geneformer.

    Extracts the dense matrix from adata.X, applies zero imputation,
    rounds to integers, clips negatives, and writes the result back to
    adata.X. Also computes ``n_counts`` and sets ``ensembl_id`` in var.

    When ``device`` is ``"cuda"``, the rounding and clipping are performed
    on GPU via torch tensors for faster throughput on large matrices.
    The result is always returned as a CPU numpy int32 array (required by
    AnnData and the downstream Geneformer tokenizer).

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with expression values in .X (sparse or dense).
    device : str
        Torch device string (``"cpu"`` or ``"cuda"``). Passed from the
        top-level pipeline so the same device flag controls everything.

    Returns
    -------
    ad.AnnData
        Modified AnnData with integer pseudo-counts in .X, ready for
        Geneformer tokenization.

    Notes
    -----
    Geneformer expects raw integer UMI counts. If the input is TPM,
    rounding introduces error — but this is the current pipeline approach.
    Verify your input data type before using this function.
    """
    import torch

    logger.info("prepare_pseudo_counts: device='{}'", device)

    if sp.issparse(adata.X):
        dense = adata.X.toarray().astype(np.float32)
    else:
        dense = np.array(adata.X, dtype=np.float32)

    # Zero imputation (CPU — fast, no large alloc benefit on GPU)
    expr_df = pd.DataFrame(dense, index=adata.obs_names,
                           columns=adata.var_names)
    expr_df = impute_zeros(expr_df)

    # Round + clip on GPU when available, fall back to numpy otherwise
    t = torch.tensor(expr_df.values, dtype=torch.float32, device=device)
    t = torch.clamp(torch.round(t), min=0)
    pseudo_counts = t.cpu().numpy().astype(np.int32)

    adata.X = pseudo_counts
    adata.var["ensembl_id"] = adata.var_names
    adata.obs["n_counts"] = adata.X.sum(axis=1)

    logger.info(
        "Pseudo-counts ready: shape={}, dtype={}, n_counts range=[{:.0f}, {:.0f}]",
        adata.shape,
        adata.X.dtype,
        adata.obs["n_counts"].min(),
        adata.obs["n_counts"].max(),
    )
    return adata


# ── 3. Geneformer tokenization ─────────────────────────────────────────────────

def tokenize_adata(
    adata: ad.AnnData,
    output_dir: str | Path,
    output_prefix: str = "geneformer_tokenized",
    attr_name_dict: dict[str, str] = TOKENIZER_ATTR_DICT,
    nproc: int = NPROC,
) -> Path:
    """Save AnnData as h5ad and run Geneformer TranscriptomeTokenizer.

    The h5ad is written to a temporary directory that is cleaned up
    automatically after tokenization. The resulting HuggingFace dataset
    is written to ``output_dir``.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with integer pseudo-counts in .X and required obs columns.
    output_dir : str or Path
        Directory where the tokenized HuggingFace dataset is written.
        Existing contents are removed before writing.
    output_prefix : str
        Filename prefix for the tokenized dataset.
    attr_name_dict : dict
        Mapping of AnnData obs column names → Geneformer token attribute names.
        Example: ``{"RNA_batch": "batch", "Diagnosis": "diagnosis"}``
    nproc : int
        Number of parallel processes for the tokenizer.

    Returns
    -------
    Path
        Path to the folder containing the tokenized HuggingFace dataset
        (the subfolder inside ``output_dir`` with ``dataset_info.json``).

    Raises
    ------
    RuntimeError
        If no HuggingFace dataset folder is found after tokenization.
    """
    from geneformer import TranscriptomeTokenizer

    output_dir = Path(output_dir)

    # Clean stale tokenizer output
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        h5ad_path = Path(tmpdir) / "cohort.h5ad"
        adata.write_h5ad(h5ad_path)
        logger.info("Saved temporary h5ad for tokenization: {}", h5ad_path)

        logger.info(
            "Initializing TranscriptomeTokenizer with attrs: {}", attr_name_dict,
        )
        tokenizer = TranscriptomeTokenizer(
            custom_attr_name_dict=attr_name_dict,
            nproc=nproc,
        )

        logger.info("Starting tokenization → '{}'", output_dir)
        tokenizer.tokenize_data(
            data_directory=str(Path(tmpdir)),
            output_directory=str(output_dir),
            output_prefix=output_prefix,
            file_format="h5ad",
        )

    # Locate the produced dataset subfolder
    dataset_folders = [
        p for p in output_dir.iterdir()
        if p.is_dir() and (p / "dataset_info.json").exists()
    ]
    if not dataset_folders:
        raise RuntimeError(
            f"Tokenization produced no HuggingFace dataset folder in {output_dir}. "
            "Check TranscriptomeTokenizer output above for errors."
        )

    dataset_path = dataset_folders[0]
    logger.info("Tokenization complete. Dataset saved to '{}'", dataset_path)
    return dataset_path
