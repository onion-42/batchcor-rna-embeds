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
    n_epochs: int = 150
    batch_size: int = 128
    lr: float = 1e-3
    lambda_adv: float = 0.1
    warmup_fraction: float = 0.3
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
