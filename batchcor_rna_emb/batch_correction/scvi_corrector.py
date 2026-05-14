"""Conditional VAE (cVAE) batch corrector for continuous embeddings.

Architecturally equivalent to scVI but with Gaussian likelihood
(scVI requires count data / NB likelihood — incompatible with embeddings).

Two variants
------------
1. **CVAECorrector** — original cVAE with KL regularisation.
   Loss: MSE_reconstruction + β · KL(q(z|x) || p(z))

2. **CVAEAdvCorrector** — cVAE + adversarial batch discriminator, *no KL*.
   Loss: MSE_reconstruction + λ_adv · CE(batch_pred, batch_true)
   Gradient Reversal Layer on the latent forces the encoder to remove
   batch information while the decoder still conditions on batch one-hot.

Architecture (common)
---------------------
Encoder:  input_dim → 512 → 256 → (μ, log σ²)   [batch-FREE]
Decoder:  (latent_dim + n_batches) → 256 → 512 → input_dim  [batch-CONDITIONED]

The encoder never sees batch labels → the latent space is batch-invariant.
The decoder receives a one-hot batch vector → can reconstruct batch-specific patterns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# PyTorch cVAE module
# ---------------------------------------------------------------------------

class _CVAEModule(nn.Module):
    """Conditional VAE with batch-conditioned decoder."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_batches: int,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # --- Encoder (batch-FREE) ---
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        self.encoder_body = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # --- Decoder (batch-CONDITIONED) ---
        dec_layers: list[nn.Module] = []
        prev = latent_dim + n_batches  # concat one-hot batch
        for h in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input → (mu, logvar). No batch info here."""
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self, z: torch.Tensor, batch_onehot: torch.Tensor
    ) -> torch.Tensor:
        """Decode latent + batch one-hot → reconstructed input."""
        z_cond = torch.cat([z, batch_onehot], dim=-1)
        return self.decoder(z_cond)

    def forward(
        self, x: torch.Tensor, batch_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward: encode → reparameterize → decode.

        Returns
        -------
        tuple
            (x_recon, mu, logvar, z)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, batch_onehot)
        return x_recon, mu, logvar, z


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CVAEConfig:
    """Configuration for cVAE training.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    hidden_dims : tuple[int, ...]
        Hidden layer sizes for encoder and decoder.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size for DataLoader.
    lr : float
        Learning rate.
    beta_max : float
        Maximum KL weight (β-VAE).
    warmup_fraction : float
        Fraction of epochs for β ramp-up from 0 to ``beta_max``.
    dropout : float
        Dropout rate in encoder/decoder.
    normalize : bool
        If True, z-score normalize input features during fit/transform.
    seed : int
        Random seed for reproducibility.
    """

    latent_dim: int = 128
    hidden_dims: tuple[int, ...] = (512, 256)
    n_epochs: int = 150
    batch_size: int = 128
    lr: float = 1e-3
    beta_max: float = 1.0
    warmup_fraction: float = 0.3
    dropout: float = 0.2
    normalize: bool = False
    seed: int = 42


@dataclass
class CVAETrainingHistory:
    """Training loss history.

    Parameters
    ----------
    total : list[float]
        Total loss per epoch.
    recon : list[float]
        Reconstruction (MSE) loss per epoch.
    kl : list[float]
        KL divergence per epoch.
    beta_schedule : list[float]
        β values per epoch.
    """

    total: list[float] = field(default_factory=list)
    recon: list[float] = field(default_factory=list)
    kl: list[float] = field(default_factory=list)
    beta_schedule: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CVAECorrector: high-level fit/transform API
# ---------------------------------------------------------------------------

class CVAECorrector:
    """Conditional VAE batch corrector for continuous embeddings.

    Follows the same API pattern as ``DANNCorrector``:
    ``fit(X_train, batch_labels) → transform(X) → latent``.

    Parameters
    ----------
    config : CVAEConfig or None
        Training configuration. Uses defaults if None.

    Attributes
    ----------
    model_ : _CVAEModule
        Trained cVAE model (set after ``fit``).
    history_ : CVAETrainingHistory
        Training loss history (set after ``fit``).
    batch_encoder_ : dict[str, int]
        Mapping from batch labels to integer codes.
    """

    def __init__(self, config: CVAEConfig | None = None) -> None:
        self.config = config or CVAEConfig()
        self.model_: _CVAEModule | None = None
        self.history_: CVAETrainingHistory | None = None
        self.batch_encoder_: dict[str, int] = {}
        self._n_batches: int = 0
        self._device: torch.device = torch.device("cpu")
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
    ) -> CVAECorrector:
        """Train cVAE on embedding matrix.

        Parameters
        ----------
        X : np.ndarray
            Input embedding matrix ``(n_samples, input_dim)``.
        batch_labels : np.ndarray
            Batch labels for each sample (str or int).

        Returns
        -------
        CVAECorrector
            Self, for chaining.
        """
        cfg = self.config
        _set_seeds(cfg.seed)

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("cVAE training on device: {}", self._device)

        # Encode batch labels
        self.batch_encoder_ = {
            lab: i for i, lab in enumerate(sorted(set(batch_labels)))
        }
        self._n_batches = len(self.batch_encoder_)
        batch_int = np.array(
            [self.batch_encoder_[b] for b in batch_labels], dtype=np.int64
        )

        # Z-score normalization (fit statistics on training data)
        if cfg.normalize:
            self._mean = X.astype(np.float64).mean(axis=0).astype(np.float32)
            self._std = X.astype(np.float64).std(axis=0).astype(np.float32)
            self._std[self._std < 1e-8] = 1.0  # avoid div-by-zero
            X = (X - self._mean) / self._std
            logger.info("Z-score normalization enabled (computed on train)")

        input_dim = X.shape[1]
        logger.info(
            "cVAE config: input_dim={}, latent_dim={}, n_batches={}, "
            "hidden={}, epochs={}, beta_max={:.2f}, normalize={}",
            input_dim, cfg.latent_dim, self._n_batches,
            cfg.hidden_dims, cfg.n_epochs, cfg.beta_max, cfg.normalize,
        )

        self.model_ = _CVAEModule(
            input_dim=input_dim,
            latent_dim=cfg.latent_dim,
            n_batches=self._n_batches,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        ).to(self._device)

        self.history_ = self._train_loop(X, batch_int)
        return self

    def transform(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Encode embeddings through the trained encoder → batch-invariant latent.

        Parameters
        ----------
        X : np.ndarray
            Input embedding matrix ``(n_samples, input_dim)``.
        batch_labels : np.ndarray or None
            Not used (encoder is batch-free). Kept for API symmetry.

        Returns
        -------
        np.ndarray
            Latent representations ``(n_samples, latent_dim)``, dtype float32.

        Raises
        ------
        RuntimeError
            If model has not been fitted.
        """
        if self.model_ is None:
            raise RuntimeError("CVAECorrector not fitted. Call .fit() first.")

        # Apply same normalization as training
        if self.config.normalize and self._mean is not None:
            X = (X - self._mean) / self._std

        self.model_.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self._device)

        latents: list[np.ndarray] = []
        bs = self.config.batch_size
        with torch.no_grad():
            for start in range(0, X_t.shape[0], bs):
                batch_x = X_t[start : start + bs]
                mu, _ = self.model_.encode(batch_x)
                latents.append(mu.cpu().numpy())

        return np.vstack(latents).astype(np.float32)

    def fit_transform(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
    ) -> np.ndarray:
        """Convenience: fit + transform in one call.

        Parameters
        ----------
        X : np.ndarray
            Input embedding matrix.
        batch_labels : np.ndarray
            Batch labels for each sample.

        Returns
        -------
        np.ndarray
            Latent representations ``(n_samples, latent_dim)``.
        """
        return self.fit(X, batch_labels).transform(X)

    # ------------------------------------------------------------------
    # Internal training loop
    # ------------------------------------------------------------------

    def _train_loop(
        self,
        X: np.ndarray,
        batch_int: np.ndarray,
    ) -> CVAETrainingHistory:
        """Internal training loop with β warm-up scheduling.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        batch_int : np.ndarray
            Integer-encoded batch labels.

        Returns
        -------
        CVAETrainingHistory
            Loss history.
        """
        cfg = self.config
        assert self.model_ is not None  # noqa: S101

        # Build one-hot batch tensor
        batch_onehot = np.zeros(
            (len(batch_int), self._n_batches), dtype=np.float32
        )
        batch_onehot[np.arange(len(batch_int)), batch_int] = 1.0

        dataset = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(batch_onehot),
        )
        loader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False
        )

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=cfg.lr)
        history = CVAETrainingHistory()
        warmup_epochs = int(cfg.n_epochs * cfg.warmup_fraction)

        for epoch in range(cfg.n_epochs):
            # β warm-up: linear ramp from 0 → beta_max
            if warmup_epochs > 0 and epoch < warmup_epochs:
                beta = cfg.beta_max * (epoch / warmup_epochs)
            else:
                beta = cfg.beta_max
            history.beta_schedule.append(beta)

            self.model_.train()
            ep_total, ep_recon, ep_kl = 0.0, 0.0, 0.0
            n_batches_seen = 0

            for x_batch, b_batch in loader:
                x_batch = x_batch.to(self._device)
                b_batch = b_batch.to(self._device)

                x_recon, mu, logvar, _ = self.model_(x_batch, b_batch)

                # Reconstruction: MSE
                loss_recon = nn.functional.mse_loss(x_recon, x_batch)
                # KL divergence: D_KL(q(z|x) || N(0,I))
                loss_kl = -0.5 * torch.mean(
                    1 + logvar - mu.pow(2) - logvar.exp()
                )
                loss = loss_recon + beta * loss_kl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ep_total += loss.item()
                ep_recon += loss_recon.item()
                ep_kl += loss_kl.item()
                n_batches_seen += 1

            # Average over mini-batches
            history.total.append(ep_total / max(n_batches_seen, 1))
            history.recon.append(ep_recon / max(n_batches_seen, 1))
            history.kl.append(ep_kl / max(n_batches_seen, 1))

            if (epoch + 1) % 25 == 0 or epoch == 0:
                logger.info(
                    "cVAE epoch {}/{}: total={:.4f}, recon={:.4f}, "
                    "kl={:.4f}, β={:.3f}",
                    epoch + 1, cfg.n_epochs,
                    history.total[-1], history.recon[-1],
                    history.kl[-1], beta,
                )

        logger.info(
            "cVAE training complete. Final: recon={:.4f}, kl={:.4f}",
            history.recon[-1], history.kl[-1],
        )
        return history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===================================================================
# Adversarial variant: cVAE + GRL batch discriminator (no KL term)
# ===================================================================

# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GradReversal(torch.autograd.Function):
    """Gradient reversal — passes input forward, negates gradient backward.

    During forward pass the tensor is unchanged.
    During backward pass the gradient is multiplied by ``-lam``, which
    pushes the encoder to *unlearn* whatever the downstream head
    (batch discriminator) is trying to predict.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.BackwardCFunction,
        grad_output: torch.Tensor,
    ) -> tuple:
        (lam,) = ctx.saved_tensors
        return -lam * grad_output, None


def _grad_reversal(x: torch.Tensor, lam: float) -> torch.Tensor:
    """Functional wrapper for the gradient reversal layer."""
    return _GradReversal.apply(x, lam)


# ---------------------------------------------------------------------------
# cVAE + Adversarial module
# ---------------------------------------------------------------------------

class _CVAEAdvModule(nn.Module):
    """cVAE with adversarial batch discriminator (no KL term).

    Architecture
    ------------
    Encoder (batch-FREE):   input → hidden → (μ, logvar) → z
    Decoder (batch-COND):   (z + batch_onehot) → hidden → input
    Discriminator (GRL):    GRL(z) → 128 → n_batches     [adversarial]

    The GRL ensures the encoder gradient *opposes* the discriminator's
    objective, so the encoder learns to hide batch identity from z.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_batches: int,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # --- Encoder (batch-FREE) ---
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        self.encoder_body = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # --- Decoder (batch-CONDITIONED) ---
        dec_layers: list[nn.Module] = []
        prev = latent_dim + n_batches
        for h in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # --- Batch discriminator (with GRL) ---
        self.batch_discriminator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_batches),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input → (mu, logvar). No batch info here."""
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self, z: torch.Tensor, batch_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent + batch one-hot → reconstructed input."""
        z_cond = torch.cat([z, batch_onehot], dim=-1)
        return self.decoder(z_cond)

    def forward(
        self,
        x: torch.Tensor,
        batch_onehot: torch.Tensor,
        lam: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward: encode → reparameterize → decode + adversarial.

        Returns
        -------
        tuple
            (x_recon, mu, logvar, z, batch_logits)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, batch_onehot)
        # Adversarial: GRL inverts gradients so encoder hides batch info
        batch_logits = self.batch_discriminator(_grad_reversal(z, lam))
        return x_recon, mu, logvar, z, batch_logits


# ---------------------------------------------------------------------------
# Config for adversarial variant
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CVAEAdvConfig:
    """Configuration for cVAE + adversarial training.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    hidden_dims : tuple[int, ...]
        Hidden layer sizes for encoder and decoder.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size for DataLoader.
    lr : float
        Learning rate.
    lambda_adv : float
        Weight for the adversarial (batch discrimination) loss.
    warmup_fraction : float
        Fraction of epochs for λ_adv ramp-up from 0 to ``lambda_adv``.
    dropout : float
        Dropout rate in encoder/decoder/discriminator.
    normalize : bool
        If True, z-score normalize input features during fit/transform.
    seed : int
        Random seed for reproducibility.
    """

    latent_dim: int = 64
    hidden_dims: tuple[int, ...] = (512, 256)
    n_epochs: int = 100
    batch_size: int = 128
    lr: float = 5e-4
    lambda_adv: float = 0.5
    warmup_fraction: float = 0.4
    dropout: float = 0.2
    normalize: bool = True
    grad_clip: float = 0.5
    seed: int = 42


@dataclass
class CVAEAdvTrainingHistory:
    """Training loss history for adversarial cVAE.

    Parameters
    ----------
    total : list[float]
        Total loss per epoch.
    recon : list[float]
        Reconstruction (MSE) loss per epoch.
    adv : list[float]
        Adversarial (CE) loss per epoch.
    lambda_schedule : list[float]
        λ_adv values per epoch.
    disc_accuracy : list[float]
        Batch discriminator accuracy per epoch (for diagnostics).
    """

    total: list[float] = field(default_factory=list)
    recon: list[float] = field(default_factory=list)
    adv: list[float] = field(default_factory=list)
    lambda_schedule: list[float] = field(default_factory=list)
    disc_accuracy: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CVAEAdvCorrector: high-level fit/transform API
# ---------------------------------------------------------------------------

class CVAEAdvCorrector:
    """cVAE + adversarial batch corrector for continuous embeddings.

    Unlike ``CVAECorrector`` that uses KL divergence to regularise the
    latent space (which can destroy biology), this variant uses a
    **batch discriminator with Gradient Reversal** to specifically target
    batch information while leaving biological structure intact.

    Loss = MSE_reconstruction + λ_adv × CrossEntropy(batch_pred, batch_true)

    Parameters
    ----------
    config : CVAEAdvConfig or None
        Training configuration. Uses defaults if None.
    """

    def __init__(self, config: CVAEAdvConfig | None = None) -> None:
        self.config = config or CVAEAdvConfig()
        self.model_: _CVAEAdvModule | None = None
        self.history_: CVAEAdvTrainingHistory | None = None
        self.batch_encoder_: dict[str, int] = {}
        self._n_batches: int = 0
        self._device: torch.device = torch.device("cpu")
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
    ) -> CVAEAdvCorrector:
        """Train cVAE + adversarial on embedding matrix.

        Parameters
        ----------
        X : np.ndarray
            Input embedding matrix ``(n_samples, input_dim)``.
        batch_labels : np.ndarray
            Batch labels for each sample (str or int).

        Returns
        -------
        CVAEAdvCorrector
            Self, for chaining.
        """
        cfg = self.config
        _set_seeds(cfg.seed)

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("cVAE-Adv training on device: {}", self._device)

        # Encode batch labels
        self.batch_encoder_ = {
            lab: i for i, lab in enumerate(sorted(set(batch_labels)))
        }
        self._n_batches = len(self.batch_encoder_)
        batch_int = np.array(
            [self.batch_encoder_[b] for b in batch_labels], dtype=np.int64
        )

        # Z-score normalization
        if cfg.normalize:
            self._mean = X.astype(np.float64).mean(axis=0).astype(np.float32)
            self._std = X.astype(np.float64).std(axis=0).astype(np.float32)
            self._std[self._std < 1e-8] = 1.0
            X = (X - self._mean) / self._std
            logger.info("Z-score normalization enabled (computed on train)")

        input_dim = X.shape[1]
        logger.info(
            "cVAE-Adv config: input_dim={}, latent_dim={}, n_batches={}, "
            "hidden={}, epochs={}, lambda_adv={:.2f}, normalize={}",
            input_dim, cfg.latent_dim, self._n_batches,
            cfg.hidden_dims, cfg.n_epochs, cfg.lambda_adv, cfg.normalize,
        )

        self.model_ = _CVAEAdvModule(
            input_dim=input_dim,
            latent_dim=cfg.latent_dim,
            n_batches=self._n_batches,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        ).to(self._device)

        self.history_ = self._train_loop(X, batch_int)
        return self

    def transform(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Encode embeddings → batch-invariant latent (mu only).

        Parameters
        ----------
        X : np.ndarray
            Input embedding matrix ``(n_samples, input_dim)``.
        batch_labels : np.ndarray or None
            Not used (encoder is batch-free). Kept for API symmetry.

        Returns
        -------
        np.ndarray
            Latent representations ``(n_samples, latent_dim)``.
        """
        if self.model_ is None:
            raise RuntimeError("CVAEAdvCorrector not fitted.")

        if self.config.normalize and self._mean is not None:
            X = (X - self._mean) / self._std

        self.model_.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self._device)

        latents: list[np.ndarray] = []
        bs = self.config.batch_size
        with torch.no_grad():
            for start in range(0, X_t.shape[0], bs):
                batch_x = X_t[start : start + bs]
                mu, _ = self.model_.encode(batch_x)
                latents.append(mu.cpu().numpy())

        return np.vstack(latents).astype(np.float32)

    def fit_transform(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
    ) -> np.ndarray:
        """Convenience: fit + transform in one call."""
        return self.fit(X, batch_labels).transform(X)

    # ------------------------------------------------------------------
    # Internal training loop (adversarial, no KL)
    # ------------------------------------------------------------------

    def _train_loop(
        self,
        X: np.ndarray,
        batch_int: np.ndarray,
    ) -> CVAEAdvTrainingHistory:
        """Train with MSE reconstruction + adversarial batch loss.

        Parameters
        ----------
        X : np.ndarray
            Input data (already z-scored if normalize=True).
        batch_int : np.ndarray
            Integer-encoded batch labels.

        Returns
        -------
        CVAEAdvTrainingHistory
            Loss history with recon, adversarial, and discriminator accuracy.
        """
        cfg = self.config
        assert self.model_ is not None  # noqa: S101

        # Build one-hot for decoder conditioning
        batch_onehot = np.zeros(
            (len(batch_int), self._n_batches), dtype=np.float32
        )
        batch_onehot[np.arange(len(batch_int)), batch_int] = 1.0

        dataset = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(batch_onehot),
            torch.from_numpy(batch_int),  # int labels for CE loss
        )
        loader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
        )

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=cfg.lr)
        recon_loss_fn = nn.MSELoss()
        adv_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        history = CVAEAdvTrainingHistory()
        warmup_epochs = int(cfg.n_epochs * cfg.warmup_fraction)

        for epoch in range(cfg.n_epochs):
            # λ_adv warm-up: linear ramp from 0 → lambda_adv
            if warmup_epochs > 0 and epoch < warmup_epochs:
                current_lambda = cfg.lambda_adv * (epoch / warmup_epochs)
            else:
                current_lambda = cfg.lambda_adv
            history.lambda_schedule.append(current_lambda)

            self.model_.train()
            ep_total, ep_recon, ep_adv = 0.0, 0.0, 0.0
            ep_correct, ep_total_samples = 0, 0
            n_minibatches = 0

            for x_mb, b_onehot, b_int in loader:
                x_mb = x_mb.to(self._device)
                b_onehot = b_onehot.to(self._device)
                b_int = b_int.to(self._device)

                x_recon, _mu, _logvar, _z, batch_logits = self.model_(
                    x_mb, b_onehot, lam=current_lambda,
                )

                # Reconstruction: MSE
                loss_recon = recon_loss_fn(x_recon, x_mb)
                # Adversarial: CE(batch_pred, batch_true)
                loss_adv = adv_loss_fn(batch_logits, b_int)
                # Total: GRL already scales gradient by -λ, so we do NOT
                # multiply loss_adv by λ again (that would give -λ² effect).
                loss = loss_recon + loss_adv

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model_.parameters(), cfg.grad_clip,
                )
                optimizer.step()

                ep_total += loss.item()
                ep_recon += loss_recon.item()
                ep_adv += loss_adv.item()
                n_minibatches += 1

                # Track discriminator accuracy (diagnostic)
                preds = batch_logits.argmax(dim=-1)
                ep_correct += (preds == b_int).sum().item()
                ep_total_samples += b_int.shape[0]

            # Average over mini-batches
            history.total.append(ep_total / max(n_minibatches, 1))
            history.recon.append(ep_recon / max(n_minibatches, 1))
            history.adv.append(ep_adv / max(n_minibatches, 1))
            disc_acc = ep_correct / max(ep_total_samples, 1)
            history.disc_accuracy.append(disc_acc)

            if (epoch + 1) % 25 == 0 or epoch == 0:
                logger.info(
                    "cVAE-Adv epoch {}/{}: total={:.4f}, recon={:.4f}, "
                    "adv={:.4f}, λ={:.3f}, disc_acc={:.1%}",
                    epoch + 1, cfg.n_epochs,
                    history.total[-1], history.recon[-1],
                    history.adv[-1], current_lambda, disc_acc,
                )

        logger.info(
            "cVAE-Adv training complete. Final: recon={:.4f}, adv={:.4f}, "
            "disc_acc={:.1%}",
            history.recon[-1], history.adv[-1], history.disc_accuracy[-1],
        )
        return history


# ===========================================================================
# Two-Optimizer Adversarial cVAE + MMD (CVAEAdv2)
# ===========================================================================

class _StrongDiscriminator(nn.Module):
    """Batch discriminator with spectral normalization.

    Deeper (2 hidden layers) and stabilized via spectral norm
    to provide a stronger adversarial signal to the encoder.

    Parameters
    ----------
    latent_dim : int
        Input dimensionality (latent space).
    n_batches : int
        Number of batch classes.
    hidden : tuple[int, ...]
        Hidden layer sizes.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        latent_dim: int,
        n_batches: int,
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = latent_dim
        for h in hidden:
            layers.extend([
                nn.utils.spectral_norm(nn.Linear(prev, h)),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, n_batches))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict batch logits from latent z."""
        return self.net(z)


@dataclass(frozen=True)
class CVAEAdv2Config:
    """Configuration for two-optimizer adversarial cVAE + optional MMD.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    hidden_dims : tuple[int, ...]
        Hidden layer sizes for encoder and decoder.
    disc_hidden : tuple[int, ...]
        Hidden layer sizes for the discriminator.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size for DataLoader.
    lr_enc : float
        Learning rate for encoder + decoder optimizer.
    lr_disc : float
        Learning rate for discriminator optimizer.
    lambda_adv : float
        Weight for adversarial loss (encoder maximizes disc CE).
    lambda_mmd : float
        Weight for MMD loss. Set >0 to enable MMD regularization.
    disc_steps : int
        Number of discriminator steps per encoder step.
    warmup_fraction : float
        Fraction of epochs for λ_adv/λ_mmd ramp-up from 0.
    dropout : float
        Dropout rate in encoder/decoder.
    disc_dropout : float
        Dropout rate in discriminator.
    normalize : bool
        If True, z-score normalize input features during fit/transform.
    grad_clip : float
        Max gradient norm for clipping.
    loss_type : str
        Loss combination: ``'adversarial'``, ``'mmd'``, or ``'both'``.
    seed : int
        Random seed for reproducibility.
    """

    latent_dim: int = 128
    hidden_dims: tuple[int, ...] = (512, 256)
    disc_hidden: tuple[int, ...] = (256, 128)
    n_epochs: int = 150
    batch_size: int = 128
    lr_enc: float = 5e-4
    lr_disc: float = 2e-4
    lambda_adv: float = 1.0
    lambda_mmd: float = 0.5
    disc_steps: int = 3
    warmup_fraction: float = 0.3
    dropout: float = 0.2
    disc_dropout: float = 0.3
    normalize: bool = True
    grad_clip: float = 1.0
    loss_type: str = "both"  # 'adversarial', 'mmd', or 'both'
    seed: int = 42


@dataclass
class CVAEAdv2History:
    """Training loss history for two-optimizer adversarial cVAE.

    Parameters
    ----------
    total_enc : list[float]
        Total encoder loss per epoch (recon - λ_adv*CE + λ_mmd*MMD).
    recon : list[float]
        Reconstruction (MSE) loss per epoch.
    adv_disc : list[float]
        Discriminator CE loss per epoch (disc objective).
    adv_enc : list[float]
        Encoder adversarial loss per epoch (encoder wants to maximize this).
    mmd : list[float]
        MMD loss per epoch (0 if disabled).
    disc_accuracy : list[float]
        Batch discriminator accuracy per epoch.
    lambda_schedule : list[float]
        λ values per epoch.
    """

    total_enc: list[float] = field(default_factory=list)
    recon: list[float] = field(default_factory=list)
    adv_disc: list[float] = field(default_factory=list)
    adv_enc: list[float] = field(default_factory=list)
    mmd: list[float] = field(default_factory=list)
    disc_accuracy: list[float] = field(default_factory=list)
    lambda_schedule: list[float] = field(default_factory=list)


class CVAEAdv2Corrector:
    """Two-optimizer adversarial cVAE + MMD for batch correction.

    Key differences from ``CVAEAdvCorrector``:
    1. **Separate optimizers** for encoder+decoder and discriminator.
    2. **Multiple disc steps** per encoder step (standard GAN practice).
    3. **Stronger discriminator** with spectral normalization.
    4. **Optional MMD loss** for stable distribution alignment.
    5. **No GRL** — encoder directly maximizes discriminator CE.

    Training loop per epoch:
    ```
    Phase 1 (disc_steps times):
        z = encoder(x).detach()     # freeze encoder
        logits = discriminator(z)
        loss_disc = CE(logits, batch)
        opt_disc.step()

    Phase 2 (1 time):
        z = encoder(x)              # gradients flow to encoder
        x_recon = decoder(z, batch_onehot)
        logits = discriminator(z)   # no detach, no GRL
        loss_enc = MSE(recon) - λ_adv * CE(logits, batch) + λ_mmd * MMD
        opt_enc.step()
    ```

    Parameters
    ----------
    config : CVAEAdv2Config or None
        Training configuration. Uses defaults if None.
    """

    def __init__(self, config: CVAEAdv2Config | None = None) -> None:
        self.config = config or CVAEAdv2Config()
        self.model_: _CVAEModule | None = None
        self.disc_: _StrongDiscriminator | None = None
        self.history_: CVAEAdv2History | None = None
        self.batch_encoder_: dict[str, int] = {}
        self._n_batches: int = 0
        self._device: torch.device = torch.device("cpu")
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
    ) -> CVAEAdv2Corrector:
        """Train two-optimizer adversarial cVAE + MMD on embedding matrix.

        Parameters
        ----------
        X : np.ndarray
            Input embedding matrix ``(n_samples, input_dim)``.
        batch_labels : np.ndarray
            Batch labels for each sample (str or int).

        Returns
        -------
        CVAEAdv2Corrector
            Self, for chaining.
        """
        cfg = self.config
        _set_seeds(cfg.seed)

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("cVAE-Adv2 training on device: {}", self._device)

        # Encode batch labels
        self.batch_encoder_ = {
            lab: i for i, lab in enumerate(sorted(set(batch_labels)))
        }
        self._n_batches = len(self.batch_encoder_)
        batch_int = np.array(
            [self.batch_encoder_[b] for b in batch_labels], dtype=np.int64
        )

        # Z-score normalization
        if cfg.normalize:
            self._mean = X.astype(np.float64).mean(axis=0).astype(np.float32)
            self._std = X.astype(np.float64).std(axis=0).astype(np.float32)
            self._std[self._std < 1e-8] = 1.0
            X = (X - self._mean) / self._std

        input_dim = X.shape[1]
        logger.info(
            "cVAE-Adv2 config: input_dim={}, latent={}, n_batches={}, "
            "loss_type='{}', disc_steps={}, lr_enc={}, lr_disc={}, "
            "λ_adv={:.2f}, λ_mmd={:.2f}",
            input_dim, cfg.latent_dim, self._n_batches,
            cfg.loss_type, cfg.disc_steps,
            cfg.lr_enc, cfg.lr_disc, cfg.lambda_adv, cfg.lambda_mmd,
        )

        # Build encoder+decoder (reuse _CVAEModule from KL variant)
        self.model_ = _CVAEModule(
            input_dim=input_dim,
            latent_dim=cfg.latent_dim,
            n_batches=self._n_batches,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        ).to(self._device)

        # Build strong discriminator
        self.disc_ = _StrongDiscriminator(
            latent_dim=cfg.latent_dim,
            n_batches=self._n_batches,
            hidden=cfg.disc_hidden,
            dropout=cfg.disc_dropout,
        ).to(self._device)

        self.history_ = self._train_loop(X, batch_int)
        return self

    def transform(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Encode embeddings → batch-invariant latent (mu only).

        Parameters
        ----------
        X : np.ndarray
            Input embedding matrix ``(n_samples, input_dim)``.
        batch_labels : np.ndarray or None
            Not used. Kept for API symmetry.

        Returns
        -------
        np.ndarray
            Latent representations ``(n_samples, latent_dim)``.
        """
        if self.model_ is None:
            raise RuntimeError("CVAEAdv2Corrector not fitted.")

        if self.config.normalize and self._mean is not None:
            X = (X - self._mean) / self._std

        self.model_.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self._device)

        latents: list[np.ndarray] = []
        bs = self.config.batch_size
        with torch.no_grad():
            for start in range(0, X_t.shape[0], bs):
                batch_x = X_t[start : start + bs]
                mu, _ = self.model_.encode(batch_x)
                latents.append(mu.cpu().numpy())

        return np.vstack(latents).astype(np.float32)

    def fit_transform(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
    ) -> np.ndarray:
        """Convenience: fit + transform in one call."""
        return self.fit(X, batch_labels).transform(X)

    # ------------------------------------------------------------------
    # Two-optimizer training loop
    # ------------------------------------------------------------------

    def _train_loop(
        self,
        X: np.ndarray,
        batch_int: np.ndarray,
    ) -> CVAEAdv2History:
        """Two-optimizer adversarial training loop.

        Phase 1: Train discriminator (disc_steps times per epoch-batch)
            - Freeze encoder, detach z
            - Minimize CE(disc(z), batch_true)

        Phase 2: Train encoder + decoder (1 step)
            - MSE reconstruction
            - MINUS λ_adv × CE(disc(z), batch_true)  → fool discriminator
            - PLUS λ_mmd × MMD(z_batch_i, z_batch_j)  → align distributions

        Parameters
        ----------
        X : np.ndarray
            Input data (already z-scored if normalize=True).
        batch_int : np.ndarray
            Integer-encoded batch labels.

        Returns
        -------
        CVAEAdv2History
            Training history.
        """
        from batchcor_rna_emb.batch_correction.mmd_loss import mmd_loss_all_pairs

        cfg = self.config
        assert self.model_ is not None
        assert self.disc_ is not None

        use_adv = cfg.loss_type in ("adversarial", "both")
        use_mmd = cfg.loss_type in ("mmd", "both")

        # Build one-hot for decoder conditioning
        batch_onehot = np.zeros(
            (len(batch_int), self._n_batches), dtype=np.float32
        )
        batch_onehot[np.arange(len(batch_int)), batch_int] = 1.0

        dataset = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(batch_onehot),
            torch.from_numpy(batch_int),
        )
        loader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
        )

        # Separate optimizers
        opt_enc = torch.optim.Adam(
            self.model_.parameters(), lr=cfg.lr_enc,
        )
        opt_disc = torch.optim.Adam(
            self.disc_.parameters(), lr=cfg.lr_disc,
        )

        recon_loss_fn = nn.MSELoss()
        adv_loss_fn = nn.CrossEntropyLoss()

        history = CVAEAdv2History()
        warmup_epochs = int(cfg.n_epochs * cfg.warmup_fraction)

        for epoch in range(cfg.n_epochs):
            # Lambda warm-up
            if warmup_epochs > 0 and epoch < warmup_epochs:
                lam_frac = epoch / warmup_epochs
            else:
                lam_frac = 1.0
            current_lambda_adv = cfg.lambda_adv * lam_frac
            current_lambda_mmd = cfg.lambda_mmd * lam_frac
            history.lambda_schedule.append(lam_frac)

            self.model_.train()
            self.disc_.train()

            ep_recon, ep_adv_disc, ep_adv_enc, ep_mmd = 0.0, 0.0, 0.0, 0.0
            ep_total_enc = 0.0
            ep_correct, ep_total_samples = 0, 0
            n_mb = 0

            for x_mb, b_onehot, b_int in loader:
                x_mb = x_mb.to(self._device)
                b_onehot = b_onehot.to(self._device)
                b_int = b_int.to(self._device)

                # ========== Phase 1: Train Discriminator ==========
                for _ in range(cfg.disc_steps):
                    with torch.no_grad():
                        mu, logvar = self.model_.encode(x_mb)
                        z = self.model_.reparameterize(mu, logvar)

                    logits = self.disc_(z)
                    loss_disc = adv_loss_fn(logits, b_int)

                    opt_disc.zero_grad()
                    loss_disc.backward()
                    nn.utils.clip_grad_norm_(
                        self.disc_.parameters(), cfg.grad_clip,
                    )
                    opt_disc.step()

                ep_adv_disc += loss_disc.item()

                # Track disc accuracy
                with torch.no_grad():
                    mu_d, lv_d = self.model_.encode(x_mb)
                    z_d = self.model_.reparameterize(mu_d, lv_d)
                    preds = self.disc_(z_d).argmax(dim=-1)
                    ep_correct += (preds == b_int).sum().item()
                    ep_total_samples += b_int.shape[0]

                # ========== Phase 2: Train Encoder + Decoder ==========
                mu, logvar = self.model_.encode(x_mb)
                z = self.model_.reparameterize(mu, logvar)
                x_recon = self.model_.decode(z, b_onehot)

                # Reconstruction loss
                loss_recon = recon_loss_fn(x_recon, x_mb)
                loss_enc = loss_recon

                # Adversarial: encoder wants discriminator to predict uniform distribution
                if use_adv:
                    logits_enc = self.disc_(z)  # no detach!
                    uniform_targets = torch.ones_like(logits_enc) / self._n_batches
                    loss_adv_enc = nn.functional.cross_entropy(logits_enc, uniform_targets)
                    loss_enc = loss_enc + current_lambda_adv * loss_adv_enc
                    ep_adv_enc += loss_adv_enc.item()

                # MMD: align batch distributions in latent space
                if use_mmd:
                    loss_mmd = mmd_loss_all_pairs(z, b_int)
                    loss_enc = loss_enc + current_lambda_mmd * loss_mmd
                    ep_mmd += loss_mmd.item()

                opt_enc.zero_grad()
                loss_enc.backward()
                nn.utils.clip_grad_norm_(
                    self.model_.parameters(), cfg.grad_clip,
                )
                opt_enc.step()

                ep_recon += loss_recon.item()
                ep_total_enc += loss_enc.item()
                n_mb += 1

            # Average over mini-batches
            history.total_enc.append(ep_total_enc / max(n_mb, 1))
            history.recon.append(ep_recon / max(n_mb, 1))
            history.adv_disc.append(ep_adv_disc / max(n_mb, 1))
            history.adv_enc.append(ep_adv_enc / max(n_mb, 1))
            history.mmd.append(ep_mmd / max(n_mb, 1))
            disc_acc = ep_correct / max(ep_total_samples, 1)
            history.disc_accuracy.append(disc_acc)

            if (epoch + 1) % 25 == 0 or epoch == 0:
                logger.info(
                    "Adv2 ep {}/{}: recon={:.4f}, disc_CE={:.4f}, "
                    "enc_adv={:.4f}, mmd={:.4f}, λ={:.2f}, "
                    "disc_acc={:.1%}",
                    epoch + 1, cfg.n_epochs,
                    history.recon[-1], history.adv_disc[-1],
                    history.adv_enc[-1], history.mmd[-1],
                    lam_frac, disc_acc,
                )

        logger.info(
            "cVAE-Adv2 training complete. Final: recon={:.4f}, "
            "disc_acc={:.1%}, loss_type='{}'",
            history.recon[-1], history.disc_accuracy[-1], cfg.loss_type,
        )
        return history
