"""Conditional VAE (cVAE) batch corrector for continuous embeddings.

Architecturally equivalent to scVI but with Gaussian likelihood
(scVI requires count data / NB likelihood — incompatible with embeddings).

Architecture
------------
Encoder:  input_dim → 512 → 256 → (μ, log σ²)   [batch-FREE]
Decoder:  (latent_dim + n_batches) → 256 → 512 → input_dim  [batch-CONDITIONED]
Loss:     MSE_reconstruction + β · KL(q(z|x) || p(z))

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

        input_dim = X.shape[1]
        logger.info(
            "cVAE config: input_dim={}, latent_dim={}, n_batches={}, "
            "hidden={}, epochs={}, beta_max={:.2f}",
            input_dim, cfg.latent_dim, self._n_batches,
            cfg.hidden_dims, cfg.n_epochs, cfg.beta_max,
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
