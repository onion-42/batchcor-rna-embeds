"""
batchcor_rna_emb/batch_correction/cae.py
=========================================
Conditional Autoencoder (cAE) for latent domain adaptation of
scGPT transformer embeddings across clinical cohorts.

Architecture
------------
  Encoder : Concat[L2-normed emb, one_hot(batch)]
            → (Linear → LayerNorm → LeakyReLU → Dropout) × n_layers
            → latent_dim  (plain Linear, no activation)

  Decoder : Concat[latent, one_hot(batch)]
            → (Linear → LayerNorm → LeakyReLU → Dropout) × n_layers
            → emb_dim     (plain Linear, no activation)

Domain-adaptation trick at inference
--------------------------------------
  Encoder receives the TRUE one-hot (identifies + strips domain signal).
  Decoder receives a ZERO vector  (no domain re-injected → domain-free emb).

Improvements over v1
---------------------
  - Input L2-normalisation before MSE loss (equalises norms across cohorts).
  - DataLoader keeps tensors on CPU; per-batch .to(device) avoids GPU OOM.
  - Seed propagated through CAEConfig for full reproducibility.
  - EMA is updated before the log line so epoch-1 shows a real EMA value.
  - Fully modern type hints (list / tuple, no typing imports).

References
----------
  Lopez et al. 2018   – scVI (conditional VAE for scRNA)
  Lotfollahi et al.   – scGen / trVAE (style-transfer correction)
  Pham et al. 2020    – BERMUDA (domain adaptation for scRNA)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# LOGGING
# =============================================================================
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | <cyan>{function}</cyan> | {message}"
    ),
    level="INFO",
)


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class CAEConfig:
    """
    All hyper-parameters for the Conditional Autoencoder.

    Parameters
    ----------
    emb_dim       : Dimensionality of the input scGPT embeddings (512 or 768).
    n_batches     : Number of distinct clinical cohorts / batch labels.
    latent_dim    : Bottleneck dimension. 128 is a strong default;
                    increase to 256 if emb_dim=768 or n_batches > 6.
    hidden_dims   : Encoder hidden widths (decoder mirrors in reverse).
                    Default [384, 256] gives a smooth compression from 512.
    dropout_p     : Dropout probability on every hidden layer.
                    0.1 is optimal for transformer-derived embeddings.
    leaky_slope   : Negative slope for LeakyReLU (DCGAN/scVI convention).
                    0.2 avoids dead neurons in deeper networks.
    lr            : AdamW initial learning rate.
    weight_decay  : AdamW L2 penalty — prevents decoder memorising batch signal.
    batch_size    : Mini-batch size for the DataLoader.
    max_epochs    : Hard ceiling on training epochs.
    patience      : EMA-based early-stopping patience in epochs.
    ema_alpha     : EMA smoothing factor (lower = smoother loss curve).
                    0.1 is robust to noisy clinical mini-cohorts.
    min_delta     : Minimum EMA improvement to reset the patience counter.
    seed          : Global RNG seed for full reproducibility.
    normalize_emb : If True, L2-normalise embeddings before training.
                    Strongly recommended when cohorts differ in scale.
    """
    emb_dim       : int       = 512
    n_batches     : int       = 6
    latent_dim    : int       = 128
    hidden_dims   : list[int] = field(default_factory=lambda: [384, 256])
    dropout_p     : float     = 0.1
    leaky_slope   : float     = 0.2
    lr            : float     = 1e-3
    weight_decay  : float     = 1e-4
    batch_size    : int       = 128
    max_epochs    : int       = 500
    patience      : int       = 30
    ema_alpha     : float     = 0.1
    min_delta     : float     = 1e-5
    seed          : int       = 42
    normalize_emb : bool      = True


# =============================================================================
# BUILDING BLOCK
# =============================================================================

def _fc_block(
    in_features  : int,
    out_features : int,
    dropout_p    : float,
    leaky_slope  : float,
    final_block  : bool = False,
) -> nn.Sequential:
    """
    One fully-connected block:
        Linear → LayerNorm → LeakyReLU(slope) → Dropout(p)

    LayerNorm over BatchNorm rationale
    ------------------------------------
    scGPT embeddings come out of a transformer that already applies LayerNorm
    internally.  BatchNorm re-centres across the batch dimension and distorts
    this geometry.  LayerNorm operates per-sample, is batch-size independent
    (safe for batch_size=1 and tiny clinical cohorts), and has been shown to
    produce more stable training on pre-normalised representations.

    final_block=True
    -----------------
    The final reconstruction (decoder output) layer uses NO activation and
    NO normalisation so the output lives in unbounded ℝ^d, matching the
    scGPT embedding space.
    """
    layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
    if not final_block:
        layers += [
            nn.LayerNorm(out_features),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Dropout(p=dropout_p),
        ]
    return nn.Sequential(*layers)


# =============================================================================
# ENCODER
# =============================================================================

class ConditionalEncoder(nn.Module):
    """
    Encoder:  Concat[emb (D), one_hot (B)]  →  latent (L)

    Input dimension  = emb_dim + n_batches
    Output dimension = latent_dim  (plain Linear, no activation)

    The one-hot batch label is concatenated so the encoder can identify and
    disentangle domain-specific variance from shared biological signal.
    Hidden layers follow _fc_block(Linear → LayerNorm → LeakyReLU → Dropout).
    The final projection to latent_dim is an unconstrained Linear.
    """

    def __init__(self, cfg: CAEConfig) -> None:
        super().__init__()
        input_dim = cfg.emb_dim + cfg.n_batches
        dims      = [input_dim] + cfg.hidden_dims

        self.hidden = nn.Sequential(*[
            _fc_block(dims[i], dims[i + 1], cfg.dropout_p, cfg.leaky_slope)
            for i in range(len(dims) - 1)
        ])
        self.projection = nn.Linear(dims[-1], cfg.latent_dim)

        logger.debug(
            f"Encoder dims: {input_dim} → {cfg.hidden_dims} → {cfg.latent_dim}"
        )

    def forward(
        self,
        emb     : torch.Tensor,   # (B, emb_dim)
        one_hot : torch.Tensor,   # (B, n_batches)
    ) -> torch.Tensor:            # (B, latent_dim)
        x = torch.cat([emb, one_hot], dim=-1)
        return self.projection(self.hidden(x))


# =============================================================================
# DECODER
# =============================================================================

class ConditionalDecoder(nn.Module):
    """
    Decoder:  Concat[latent (L), one_hot (B)]  →  reconstructed emb (D)

    Mirrored architecture of the Encoder (hidden_dims traversed in reverse).
    The one-hot is re-injected during training so the decoder learns to add
    domain signal back; passing zeros at inference strips that signal.

    Input dimension  = latent_dim + n_batches
    Output dimension = emb_dim  (final_block=True → no activation / norm)
    """

    def __init__(self, cfg: CAEConfig) -> None:
        super().__init__()
        input_dim = cfg.latent_dim + cfg.n_batches
        rev_dims  = list(reversed(cfg.hidden_dims))
        dims      = [input_dim] + rev_dims

        self.hidden = nn.Sequential(*[
            _fc_block(dims[i], dims[i + 1], cfg.dropout_p, cfg.leaky_slope)
            for i in range(len(dims) - 1)
        ])
        self.reconstruction = _fc_block(
            dims[-1], cfg.emb_dim,
            cfg.dropout_p, cfg.leaky_slope,
            final_block=True,
        )

        logger.debug(
            f"Decoder dims: {input_dim} → {rev_dims} → {cfg.emb_dim}"
        )

    def forward(
        self,
        latent  : torch.Tensor,   # (B, latent_dim)
        one_hot : torch.Tensor,   # (B, n_batches)  – zeros at inference
    ) -> torch.Tensor:            # (B, emb_dim)
        x = torch.cat([latent, one_hot], dim=-1)
        return self.reconstruction(self.hidden(x))


# =============================================================================
# CONDITIONAL AUTOENCODER
# =============================================================================

class ConditionalAutoencoder(nn.Module):
    """
    Assembled cAE owning both Encoder and Decoder.

    forward()          – used during training (true one-hot → enc AND dec).
    encode() / decode() – called separately during inference.
    """

    def __init__(self, cfg: CAEConfig) -> None:
        super().__init__()
        self.cfg     = cfg
        self.encoder = ConditionalEncoder(cfg)
        self.decoder = ConditionalDecoder(cfg)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"ConditionalAutoencoder | emb_dim={cfg.emb_dim} | "
            f"n_batches={cfg.n_batches} | latent_dim={cfg.latent_dim} | "
            f"hidden_dims={cfg.hidden_dims} | trainable_params={n_params:,}"
        )

    def forward(
        self,
        emb     : torch.Tensor,
        one_hot : torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(self.encoder(emb, one_hot), one_hot)

    def encode(self, emb: torch.Tensor, one_hot: torch.Tensor) -> torch.Tensor:
        return self.encoder(emb, one_hot)

    def decode(self, latent: torch.Tensor, one_hot: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent, one_hot)


# =============================================================================
# HELPERS
# =============================================================================

def _detect_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU detected — running on CPU.")
    return device


def _set_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch RNG seeds for full reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug(f"Global RNG seed set to {seed}.")


def _l2_normalize(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Row-wise L2 normalisation with safe division.

    Returns the normalised array and the per-row norms so the caller can
    optionally rescale the corrected embeddings back to the original scale.

    Why normalise?
    --------------
    scGPT embeddings from different cohorts may have very different L2 norms
    depending on sequencing depth, library size, and preprocessing choices.
    MSE loss on unnormalised embeddings is dominated by the scale differences
    rather than the biological structure, causing the cAE to learn a trivial
    scale-matching function instead of domain adaptation.
    """
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # safe divide
    return (emb / norms).astype(np.float32), norms.astype(np.float32)


def _make_one_hot(
    batch_indices : np.ndarray,
    n_batches     : int,
) -> torch.Tensor:
    """
    Convert integer batch labels → float32 one-hot tensor (CPU).

    Validates range so out-of-range labels crash with a clear message instead
    of silently indexing into the wrong column.
    """
    idx = np.asarray(batch_indices, dtype=np.int64)
    if len(idx) == 0:
        return torch.zeros(0, n_batches, dtype=torch.float32)
    if idx.min() < 0 or idx.max() >= n_batches:
        raise ValueError(
            f"batch_indices must be in [0, {n_batches - 1}]. "
            f"Got range [{idx.min()}, {idx.max()}]."
        )
    oh = torch.zeros(len(idx), n_batches, dtype=torch.float32)
    oh[torch.arange(len(idx)), torch.from_numpy(idx)] = 1.0
    return oh  # stays on CPU; moved per-batch in the training loop


def _validate_batch_completeness(
    batch_indices : np.ndarray,
    n_batches     : int,
) -> None:
    """Every label in ``0 … n_batches-1`` must appear at least once in training."""
    seen   = set(np.unique(batch_indices).tolist())
    expect = set(range(n_batches))
    absent = expect - seen
    if absent:
        raise ValueError(
            f"Batch indices {sorted(absent)} never appear in training data "
            f"but n_batches={n_batches} implies they should.\n"
            "Pass explicit batch labels or set cfg.n_batches to the number of "
            "cohorts actually present."
        )


def _build_dataloader(
    embeddings    : np.ndarray,
    batch_indices : np.ndarray,
    n_batches     : int,
    batch_size    : int,
    shuffle       : bool = True,
    seed          : int  = 42,
) -> DataLoader:
    """
    Build a CPU-resident DataLoader.

    Keeping tensors on CPU and calling .to(device) per-batch is the only
    safe strategy for large cohorts on GPUs with < 16 GB VRAM.  Moving all
    data to the GPU upfront can silently OOM when N_patients × emb_dim
    approaches the device memory limit.
    """
    emb_t = torch.from_numpy(embeddings.astype(np.float32))
    oh_t  = _make_one_hot(batch_indices, n_batches)
    ds    = TensorDataset(emb_t, oh_t)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        drop_last   = False,
        pin_memory  = torch.cuda.is_available(),
        num_workers = 0,          # 0 = main process; avoids Windows/MPS issues
        generator   = generator if shuffle else None,
    )


# =============================================================================
# EARLY STOPPING  (EMA-based, training-loss only)
# =============================================================================

class _EarlyStopping:
    """
    Monitors an Exponential Moving Average (EMA) of training loss.

    ``best_state`` is initialised from the untrained model so ``restore_best``
    always reloads valid weights (even if the first epoch was the only
    improvement).
    """

    def __init__(
        self,
        patience  : int,
        min_delta : float,
        alpha     : float,
        model     : nn.Module,
    ) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.alpha      = alpha
        self.ema        : Optional[float] = None
        self.best_ema   : float           = float("inf")
        self.counter    : int             = 0
        self.best_state : dict[str, torch.Tensor] = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    def update(self, loss: float, model: nn.Module) -> bool:
        """
        Update EMA first, then evaluate patience.
        Returns True when training should stop.

        Note: EMA is updated BEFORE logging so the first epoch reports a
        real EMA value rather than the 'init' placeholder.
        """
        self.ema = (
            loss
            if self.ema is None
            else self.alpha * loss + (1.0 - self.alpha) * self.ema
        )

        if self.ema < self.best_ema - self.min_delta:
            self.best_ema   = self.ema
            self.counter    = 0
            # Save best weights to CPU to avoid occupying extra VRAM
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        model.load_state_dict(self.best_state)
        logger.info(
            f"Best weights restored (best EMA loss = {self.best_ema:.6f})."
        )


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_cae(
    embeddings    : np.ndarray,
    batch_indices : np.ndarray,
    cfg           : Optional[CAEConfig] = None,
    device        : Optional[torch.device] = None,
) -> ConditionalAutoencoder:
    """
    Train a ConditionalAutoencoder on pooled embeddings from all cohorts.

    Parameters
    ----------
    embeddings    : float32 (N, emb_dim) — concatenated scGPT embeddings.
    batch_indices : int64  (N,)          — cohort label per patient [0, B-1].
    cfg           : CAEConfig. Auto-inferred from data shape if None.
    device        : torch.device. Auto-detected (cuda > cpu) if None.

    Returns
    -------
    Trained ConditionalAutoencoder in eval() mode.

    Raises
    ------
    ValueError   – shape mismatches or out-of-range batch labels.
    RuntimeError – CUDA OOM: reduce cfg.batch_size.
    """
    # ── Input validation ──────────────────────────────────────────────────────
    embeddings    = np.asarray(embeddings,    dtype=np.float32)
    batch_indices = np.asarray(batch_indices, dtype=np.int64)

    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-D (N, emb_dim), got {embeddings.shape}."
        )
    if batch_indices.ndim != 1 or len(batch_indices) != len(embeddings):
        raise ValueError(
            f"batch_indices must be 1-D with length {len(embeddings)}, "
            f"got {batch_indices.shape}."
        )

    n_samples, emb_dim = embeddings.shape
    n_batches_inferred = int(batch_indices.max()) + 1

    # ── Config & device ───────────────────────────────────────────────────────
    if cfg is None:
        cfg = CAEConfig(emb_dim=emb_dim, n_batches=n_batches_inferred)
        logger.info(
            f"CAEConfig inferred: emb_dim={emb_dim}, n_batches={n_batches_inferred}"
        )
    elif cfg.n_batches != n_batches_inferred:
        raise ValueError(
            f"cfg.n_batches={cfg.n_batches} but data implies "
            f"n_batches={n_batches_inferred} (max batch index "
            f"{int(batch_indices.max())})."
        )

    if cfg.emb_dim != emb_dim:
        raise ValueError(
            f"cfg.emb_dim={cfg.emb_dim} does not match embedding dim={emb_dim}."
        )

    _validate_batch_completeness(batch_indices, cfg.n_batches)

    if device is None:
        device = _detect_device()

    # ── Reproducibility ───────────────────────────────────────────────────────
    _set_seeds(cfg.seed)

    # ── L2 normalisation (FIX: equalise embedding norms across cohorts) ───────
    if cfg.normalize_emb:
        logger.info(
            "L2-normalising embeddings before training "
            "(cfg.normalize_emb=True)."
        )
        embeddings, _norms = _l2_normalize(embeddings)
    else:
        logger.info("Skipping L2 normalisation (cfg.normalize_emb=False).")

    logger.info(
        f"Training cAE | n_samples={n_samples:,} | emb_dim={emb_dim} | "
        f"n_batches={cfg.n_batches} | latent_dim={cfg.latent_dim} | "
        f"hidden_dims={cfg.hidden_dims} | max_epochs={cfg.max_epochs} | "
        f"patience={cfg.patience} | seed={cfg.seed}"
    )

    # ── Model, optimiser, loss, scheduler ────────────────────────────────────
    model     = ConditionalAutoencoder(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # Cosine annealing restarts momentum across epochs — better than fixed LR
    # for transformer embeddings which can have flat loss landscapes.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs, eta_min=cfg.lr * 0.01
    )
    criterion = nn.MSELoss()

    # ── DataLoader — CPU-resident (FIX: avoids GPU OOM on large cohorts) ─────
    loader = _build_dataloader(
        embeddings, batch_indices,
        n_batches  = cfg.n_batches,
        batch_size = cfg.batch_size,
        shuffle    = True,
        seed       = cfg.seed,
    )

    # ── Early stopping ────────────────────────────────────────────────────────
    stopper = _EarlyStopping(
        patience  = cfg.patience,
        min_delta = cfg.min_delta,
        alpha     = cfg.ema_alpha,
        model     = model,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()

    for epoch in range(1, cfg.max_epochs + 1):
        epoch_loss = 0.0

        for emb_batch, oh_batch in loader:
            # Move to device per-batch (CPU DataLoader → GPU batch transfer)
            emb_batch = emb_batch.to(device, non_blocking=True)
            oh_batch  = oh_batch.to(device,  non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            recon = model(emb_batch, oh_batch)
            loss  = criterion(recon, emb_batch)
            loss.backward()

            # Gradient clipping: prevents exploding gradients in deep nets
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(emb_batch)

        epoch_loss /= n_samples
        scheduler.step()

        # FIX: update EMA *before* logging so epoch 1 shows a real EMA value
        should_stop = stopper.update(epoch_loss, model)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:>4}/{cfg.max_epochs} | "
                f"loss={epoch_loss:.6f} | "
                f"EMA={stopper.ema:.6f} | "
                f"LR={scheduler.get_last_lr()[0]:.2e} | "
                f"patience={stopper.counter}/{cfg.patience}"
            )

        if should_stop:
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(best EMA={stopper.best_ema:.6f})."
            )
            break
    else:
        logger.info(
            f"Training completed {cfg.max_epochs} epochs "
            f"(best EMA={stopper.best_ema:.6f})."
        )

    stopper.restore_best(model)
    model.eval()
    logger.info("cAE training complete. Model in eval mode.")
    return model


# =============================================================================
# INFERENCE — BATCH CORRECTION
# =============================================================================

def correct_embeddings(
    model         : ConditionalAutoencoder,
    embeddings    : np.ndarray,
    batch_indices : np.ndarray,
    batch_size    : int = 512,
    device        : Optional[torch.device] = None,
) -> np.ndarray:
    """
    Produce batch-corrected embeddings for all patients.

    The domain-adaptation trick
    ---------------------------
    Training:
        latent = Encoder( emb,   true_one_hot )
        recon  = Decoder( latent, true_one_hot )   ← MSE vs emb

    Inference:
        latent    = Encoder( emb,   true_one_hot )   # strip domain
        corrected = Decoder( latent, ZEROS        )   # no domain re-injected

    Because the Decoder was trained to use the one-hot to restore domain
    signal, feeding zeros forces it to output a domain-neutral reconstruction
    — the shared biological state across all cohorts.

    L2 normalisation
    ----------------
    If cfg.normalize_emb is True the same normalisation applied during
    training is re-applied here so the corrected embeddings live in the same
    unit-sphere as the training targets.

    Parameters
    ----------
    model         : Trained ConditionalAutoencoder (eval mode).
    embeddings    : float32 (N, emb_dim) raw scGPT embeddings.
    batch_indices : int64  (N,) cohort labels matching training labels.
    batch_size    : Inference batch size (can safely be 4× training size).
    device        : Auto-detected if None.

    Returns
    -------
    corrected : float32 numpy array of shape (N, emb_dim).
    """
    embeddings    = np.asarray(embeddings,    dtype=np.float32)
    batch_indices = np.asarray(batch_indices, dtype=np.int64)

    if device is None:
        device = _detect_device()

    model = model.to(device)
    model.eval()

    cfg = model.cfg

    # Apply same normalisation used during training
    if cfg.normalize_emb:
        embeddings, _ = _l2_normalize(embeddings)

    n_samples = len(embeddings)
    corrected = np.empty_like(embeddings)

    logger.info(
        f"Correcting {n_samples:,} embeddings | "
        f"emb_dim={cfg.emb_dim} | latent_dim={cfg.latent_dim}"
    )

    # CPU DataLoader — same strategy as training
    loader = _build_dataloader(
        embeddings, batch_indices,
        n_batches  = cfg.n_batches,
        batch_size = batch_size,
        shuffle    = False,   # order MUST be preserved for index alignment
    )

    ptr = 0
    with torch.no_grad():
        for emb_batch, oh_batch in loader:
            bsz      = emb_batch.shape[0]
            emb_gpu  = emb_batch.to(device, non_blocking=True)
            oh_gpu   = oh_batch.to(device,  non_blocking=True)

            # Encode with TRUE one-hot → strip domain signal
            latent  = model.encode(emb_gpu, oh_gpu)
            # Decode with ZERO vector → no domain conditioning
            neutral = torch.zeros_like(oh_gpu)
            recon   = model.decode(latent, neutral)

            corrected[ptr : ptr + bsz] = recon.cpu().numpy()
            ptr += bsz

    if ptr != n_samples:
        raise RuntimeError(
            f"Inference wrote {ptr} rows but expected {n_samples}. "
            "This is a bug — please open an issue."
        )

    logger.info(
        f"Batch correction complete | "
        f"shape={corrected.shape} | dtype={corrected.dtype}"
    )
    return corrected