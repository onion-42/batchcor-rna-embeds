"""Domain Adversarial Neural Network (DANN) for batch correction.

Architecture:
  Encoder → latent representation
  Batch discriminator (with GRL) → adversarial batch prediction
  Bio classifier (optional) → preserve biological signal (e.g. diagnosis)

Loss: L_recon + lambda_adv * L_adv - lambda_bio * L_bio
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GradientReversal(torch.autograd.Function):
    """Gradient reversal layer — scales gradient by -lambda during backward."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(ctx: torch.autograd.function.BackwardCFunction, grad_output: torch.Tensor) -> tuple:
        (lam,) = ctx.saved_tensors
        return -lam * grad_output, None


def _gradient_reversal(x: torch.Tensor, lam: float) -> torch.Tensor:
    return _GradientReversal.apply(x, lam)


# ---------------------------------------------------------------------------
# DANN model
# ---------------------------------------------------------------------------

class _DANNModel(nn.Module):
    """DANN PyTorch module with encoder, batch discriminator, and optional bio head."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_batches: int,
        n_bio_classes: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        self.batch_discriminator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_batches),
        )

        self.bio_classifier: nn.Module | None = None
        if n_bio_classes > 0:
            self.bio_classifier = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, n_bio_classes),
            )

    def forward(
        self,
        x: torch.Tensor,
        lam: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings (batch_size, input_dim).
        lam : float
            GRL lambda for adversarial gradient scaling.

        Returns
        -------
        tuple
            (latent, batch_logits, bio_logits_or_None)
        """
        latent = self.encoder(x)
        reversed_latent = _gradient_reversal(latent, lam)
        batch_logits = self.batch_discriminator(reversed_latent)

        bio_logits = None
        if self.bio_classifier is not None:
            bio_logits = self.bio_classifier(latent)

        return latent, batch_logits, bio_logits


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DANNConfig:
    """Configuration for DANN training.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate.
    lambda_adv : float
        Final adversarial loss weight.
    lambda_bio : float
        Biological preservation loss weight.
    warmup_fraction : float
        Fraction of epochs for progressive lambda warmup (0 to lambda_adv).
    dropout : float
        Dropout rate in encoder.
    seed : int
        Random seed for reproducibility.
    """

    latent_dim: int = 256
    n_epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    lambda_adv: float = 1.0
    lambda_bio: float = 0.5
    warmup_fraction: float = 0.3
    dropout: float = 0.3
    seed: int = 42


@dataclass
class DANNTrainingHistory:
    """Training loss history.

    Parameters
    ----------
    total : list[float]
        Total loss per epoch.
    recon : list[float]
        Reconstruction loss per epoch.
    adv : list[float]
        Adversarial loss per epoch.
    bio : list[float]
        Bio classifier loss per epoch.
    lambda_schedule : list[float]
        Lambda values per epoch.
    """

    total: list[float] = field(default_factory=list)
    recon: list[float] = field(default_factory=list)
    adv: list[float] = field(default_factory=list)
    bio: list[float] = field(default_factory=list)
    lambda_schedule: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DANNCorrector: high-level fit/transform API
# ---------------------------------------------------------------------------

class DANNCorrector:
    """Domain Adversarial Neural Network for batch correction of embeddings.

    Parameters
    ----------
    config : DANNConfig
        Training configuration.

    Attributes
    ----------
    model_ : _DANNModel
        Trained DANN model (set after ``fit``).
    history_ : DANNTrainingHistory
        Training loss history (set after ``fit``).
    batch_encoder_ : dict[str, int]
        Mapping from batch labels to integer codes.
    bio_encoder_ : dict[str, int] or None
        Mapping from bio labels to integer codes.
    """

    def __init__(self, config: DANNConfig | None = None) -> None:
        self.config = config or DANNConfig()
        self.model_: _DANNModel | None = None
        self.history_: DANNTrainingHistory | None = None
        self.batch_encoder_: dict[str, int] = {}
        self.bio_encoder_: dict[str, int] | None = None
        self._device: torch.device = torch.device("cpu")

    def fit(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
        bio_labels: np.ndarray | None = None,
    ) -> DANNCorrector:
        """
        Train DANN on embedding matrix.

        Parameters
        ----------
        X : np.ndarray
            Input embedding matrix of shape ``(n_samples, n_features)``.
        batch_labels : np.ndarray
            Batch labels for each sample.
        bio_labels : np.ndarray or None
            Biological labels (e.g. diagnosis) for optional bio classifier.

        Returns
        -------
        DANNCorrector
            Self, for chaining.

        Raises
        ------
        ValueError
            If shapes are inconsistent.
        """
        cfg = self.config
        _set_seeds(cfg.seed)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("DANN training on device: {}", self._device)

        # encode labels
        self.batch_encoder_ = {lab: i for i, lab in enumerate(np.unique(batch_labels))}
        batch_int = np.array([self.batch_encoder_[b] for b in batch_labels], dtype=np.int64)
        n_batches = len(self.batch_encoder_)

        bio_int: np.ndarray | None = None
        n_bio = 0
        if bio_labels is not None:
            self.bio_encoder_ = {lab: i for i, lab in enumerate(np.unique(bio_labels))}
            bio_int = np.array([self.bio_encoder_[b] for b in bio_labels], dtype=np.int64)
            n_bio = len(self.bio_encoder_)

        input_dim = X.shape[1]
        self.model_ = _DANNModel(
            input_dim=input_dim,
            latent_dim=cfg.latent_dim,
            n_batches=n_batches,
            n_bio_classes=n_bio,
            dropout=cfg.dropout,
        ).to(self._device)

        self.history_ = self._train_loop(X, batch_int, bio_int)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform embeddings through the trained encoder.

        Parameters
        ----------
        X : np.ndarray
            Input embedding matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Latent representations of shape ``(n_samples, latent_dim)``.

        Raises
        ------
        RuntimeError
            If model has not been fitted.
        """
        if self.model_ is None:
            raise RuntimeError("DANNCorrector not fitted. Call .fit() first.")

        self.model_.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self._device)

        latents = []
        bs = self.config.batch_size
        with torch.no_grad():
            for start in range(0, X_t.shape[0], bs):
                batch_x = X_t[start:start + bs]
                latent, _, _ = self.model_(batch_x, lam=0.0)
                latents.append(latent.cpu().numpy())

        return np.vstack(latents).astype(np.float32)

    def _train_loop(
        self,
        X: np.ndarray,
        batch_int: np.ndarray,
        bio_int: np.ndarray | None,
    ) -> DANNTrainingHistory:
        """
        Internal training loop with progressive lambda scheduling.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        batch_int : np.ndarray
            Integer-encoded batch labels.
        bio_int : np.ndarray or None
            Integer-encoded bio labels.

        Returns
        -------
        DANNTrainingHistory
            Loss history.
        """
        cfg = self.config
        assert self.model_ is not None  # noqa: S101

        tensors = [
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(batch_int),
        ]
        if bio_int is not None:
            tensors.append(torch.from_numpy(bio_int))

        dataset = TensorDataset(*tensors)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=cfg.lr)
        recon_loss_fn = nn.MSELoss()
        adv_loss_fn = nn.CrossEntropyLoss()
        bio_loss_fn = nn.CrossEntropyLoss() if bio_int is not None else None

        history = DANNTrainingHistory()
        warmup_epochs = int(cfg.n_epochs * cfg.warmup_fraction)

        for epoch in range(cfg.n_epochs):
            # progressive lambda: linear ramp from 0 to lambda_adv over warmup
            if warmup_epochs > 0 and epoch < warmup_epochs:
                current_lambda = cfg.lambda_adv * (epoch / warmup_epochs)
            else:
                current_lambda = cfg.lambda_adv
            history.lambda_schedule.append(current_lambda)

            self.model_.train()
            epoch_total, epoch_recon, epoch_adv, epoch_bio = 0.0, 0.0, 0.0, 0.0
            n_batches_seen = 0

            for batch_data in loader:
                x = batch_data[0].to(self._device)
                b = batch_data[1].to(self._device)
                bio = batch_data[2].to(self._device) if len(batch_data) > 2 else None

                latent, batch_logits, bio_logits = self.model_(x, lam=current_lambda)

                # reconstruction loss (identity through encoder)
                loss_recon = recon_loss_fn(latent, x[:, :latent.shape[1]])
                # adversarial: make batch indistinguishable
                loss_adv = adv_loss_fn(batch_logits, b)
                # total: minimize recon + lambda*adv (GRL handles sign flip)
                loss = loss_recon + current_lambda * loss_adv

                loss_bio_val = 0.0
                if bio_logits is not None and bio is not None:
                    loss_bio = bio_loss_fn(bio_logits, bio)
                    loss = loss - cfg.lambda_bio * loss_bio
                    loss_bio_val = loss_bio.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_total += loss.item()
                epoch_recon += loss_recon.item()
                epoch_adv += loss_adv.item()
                epoch_bio += loss_bio_val
                n_batches_seen += 1

            # average over mini-batches
            history.total.append(epoch_total / max(n_batches_seen, 1))
            history.recon.append(epoch_recon / max(n_batches_seen, 1))
            history.adv.append(epoch_adv / max(n_batches_seen, 1))
            history.bio.append(epoch_bio / max(n_batches_seen, 1))

            if (epoch + 1) % 20 == 0 or epoch == 0:
                logger.info(
                    "DANN epoch {}/{}: total={:.4f}, recon={:.4f}, adv={:.4f}, bio={:.4f}, lambda={:.3f}",
                    epoch + 1, cfg.n_epochs,
                    history.total[-1], history.recon[-1],
                    history.adv[-1], history.bio[-1], current_lambda,
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
