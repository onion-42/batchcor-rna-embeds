"""Tests for scGPT preprocessing helpers (synthetic data only)."""

from __future__ import annotations

import numpy as np
import torch
from anndata import AnnData

from batchcor_rna_emb.modeling.scgpt_embeddings import (
    N_BINS,
    bin_expression,
    build_model_inputs,
    preprocess_adata,
)


def test_bin_expression_shape_and_zero_reserved():
    expr = np.array(
        [[0.0, 0.5, 2.0], [1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    b = bin_expression(expr, n_bins=11)
    assert b.shape == expr.shape
    assert b.dtype == np.int64
    assert np.all((b >= 0) & (b < 11))
    assert b[0, 0] == 0
    assert np.all(b[1] >= 0)


def test_bin_expression_single_nonzero_value_row():
    expr = np.array([[0.0, 3.0, 3.0]], dtype=np.float32)
    b = bin_expression(expr, n_bins=51)
    assert b[0, 0] == 0
    assert np.all(b[0, 1:] == 1)


def test_build_model_inputs_layout():
    n_cells, n_genes = 4, 7
    pad_id = 12345
    expr_binned = np.zeros((n_cells, n_genes), dtype=np.int64)
    expr_binned[:, :3] = np.arange(1, 4)
    gene_ids = np.arange(100, 100 + n_genes, dtype=np.int64)

    gene_tokens, expr_values, key_pad_mask = build_model_inputs(
        expr_binned, gene_ids, pad_id=pad_id, n_bins=N_BINS
    )
    seq_len = n_genes + 1
    assert gene_tokens.shape == (n_cells, seq_len)
    assert expr_values.shape == (n_cells, seq_len)
    assert key_pad_mask.shape == (n_cells, seq_len)
    assert bool((gene_tokens[:, 0] == pad_id).all())
    assert bool((expr_values[:, 0] == N_BINS).all())
    assert not key_pad_mask.any()
    expected_genes = torch.tensor(gene_ids, dtype=torch.long).unsqueeze(0).expand(n_cells, -1)
    assert torch.equal(gene_tokens[:, 1:], expected_genes)


def test_preprocess_adata_skips_norm_when_log_normalized():
    # max <= 14 triggers "already log-normalised" path
    X = np.random.RandomState(0).rand(20, 5).astype(np.float32) * 3.0
    adata = AnnData(X=X)
    out, raw_counts = preprocess_adata(adata)
    assert raw_counts is None
    assert out.n_obs == adata.n_obs
    assert out.n_vars == adata.n_vars


def test_preprocess_adata_detects_raw_like_scale():
    X = np.full((15, 4), 50.0, dtype=np.float32)
    adata = AnnData(X=X)
    out, raw_counts = preprocess_adata(adata)
    assert raw_counts is not None
    assert raw_counts.shape == (15, 4)
    # ScanPy normalisation should shrink typical magnitudes
    Xm = out.X.toarray() if hasattr(out.X, "toarray") else np.asarray(out.X)
    assert float(np.asarray(Xm).max()) < 20.0
