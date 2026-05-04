"""Unit tests for ``ConditionalAutoencoder`` (no real embeddings or GPU required)."""

from __future__ import annotations

import torch

from batchcor_rna_emb.batch_correction.cae import CAEConfig, ConditionalAutoencoder


def _one_hot_batch(batch_idx: int, n_batches: int, batch_size: int) -> torch.Tensor:
    oh = torch.zeros(batch_size, n_batches, dtype=torch.float32)
    oh[torch.arange(batch_size), batch_idx] = 1.0
    return oh


def test_cae_initialization_and_param_count():
    cfg = CAEConfig(
        emb_dim=32,
        n_batches=3,
        latent_dim=8,
        hidden_dims=[24, 16],
        max_epochs=1,
    )
    model = ConditionalAutoencoder(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert n > 0
    assert isinstance(model.encoder, torch.nn.Module)
    assert isinstance(model.decoder, torch.nn.Module)


def test_cae_forward_encode_decode_shapes():
    cfg = CAEConfig(
        emb_dim=32,
        n_batches=4,
        latent_dim=8,
        hidden_dims=[20],
        max_epochs=1,
    )
    model = ConditionalAutoencoder(cfg)
    model.eval()
    B = 5
    emb = torch.randn(B, cfg.emb_dim)
    one_hot = _one_hot_batch(batch_idx=1, n_batches=cfg.n_batches, batch_size=B)

    recon = model(emb, one_hot)
    assert recon.shape == (B, cfg.emb_dim)

    z = model.encode(emb, one_hot)
    assert z.shape == (B, cfg.latent_dim)

    out_dec = model.decode(z, one_hot)
    assert out_dec.shape == (B, cfg.emb_dim)


def test_cae_forward_finite_and_gradflow():
    cfg = CAEConfig(emb_dim=16, n_batches=2, latent_dim=4, hidden_dims=[12], max_epochs=1)
    model = ConditionalAutoencoder(cfg)
    emb = torch.randn(3, cfg.emb_dim, requires_grad=True)
    oh = _one_hot_batch(0, cfg.n_batches, 3)
    y = model(emb, oh)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert emb.grad is not None and torch.isfinite(emb.grad).all()
