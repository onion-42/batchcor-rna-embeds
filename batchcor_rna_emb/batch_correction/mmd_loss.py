"""Maximum Mean Discrepancy (MMD) loss for batch correction.

Multi-scale RBF kernel MMD provides a stable, non-adversarial alternative
to GRL-based domain adaptation. It aligns latent distributions of different
batches without requiring a discriminator or min-max optimization.

References
----------
- Gretton et al., "A Kernel Two-Sample Test", JMLR 2012
- Li et al., "MMD GAN: Towards Deeper Understanding of Moment Matching Network", NeurIPS 2017
"""
from __future__ import annotations

import torch


def _rbf_kernel_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    gammas: list[float],
) -> torch.Tensor:
    """Compute sum of RBF kernels at multiple bandwidths.

    Parameters
    ----------
    x : torch.Tensor
        First sample, shape ``(n, d)``.
    y : torch.Tensor
        Second sample, shape ``(m, d)``.
    gammas : list[float]
        Bandwidth parameters for multi-scale RBF.

    Returns
    -------
    torch.Tensor
        Kernel matrix of shape ``(n, m)``.
    """
    dist_sq = torch.cdist(x, y, p=2.0).pow(2)
    K = torch.zeros_like(dist_sq)
    for g in gammas:
        K = K + torch.exp(-g * dist_sq)
    return K


def mmd_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    gammas: list[float] | None = None,
) -> torch.Tensor:
    """Compute unbiased MMD² estimate with multi-scale RBF kernel.

    Parameters
    ----------
    x : torch.Tensor
        Samples from distribution P, shape ``(n, d)``.
    y : torch.Tensor
        Samples from distribution Q, shape ``(m, d)``.
    gammas : list[float] or None
        RBF bandwidth parameters. If None, uses ``[0.001, 0.01, 0.1, 1, 10]``.

    Returns
    -------
    torch.Tensor
        Scalar MMD² estimate (non-negative for well-separated distributions).
    """
    if gammas is None:
        gammas = [0.001, 0.01, 0.1, 1.0, 10.0]

    Kxx = _rbf_kernel_matrix(x, x, gammas)
    Kyy = _rbf_kernel_matrix(y, y, gammas)
    Kxy = _rbf_kernel_matrix(x, y, gammas)

    # Unbiased estimator: exclude diagonal for xx and yy
    n = x.shape[0]
    m = y.shape[0]

    # Use mean of off-diagonal for unbiased estimate
    mask_xx = 1.0 - torch.eye(n, device=x.device)
    mask_yy = 1.0 - torch.eye(m, device=y.device)

    mmd = (
        (Kxx * mask_xx).sum() / max(n * (n - 1), 1)
        + (Kyy * mask_yy).sum() / max(m * (m - 1), 1)
        - 2.0 * Kxy.mean()
    )
    return mmd


def mmd_loss_all_pairs(
    z: torch.Tensor,
    batch_labels: torch.Tensor,
    gammas: list[float] | None = None,
) -> torch.Tensor:
    """Compute mean MMD² across all pairs of batches.

    Parameters
    ----------
    z : torch.Tensor
        Latent representations, shape ``(n_total, latent_dim)``.
    batch_labels : torch.Tensor
        Integer batch labels, shape ``(n_total,)``.
    gammas : list[float] or None
        RBF bandwidth parameters.

    Returns
    -------
    torch.Tensor
        Mean MMD² across all batch pairs. Returns 0 if fewer than 2 batches.
    """
    unique_batches = batch_labels.unique()
    n_batches = len(unique_batches)

    if n_batches < 2:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    total_mmd = torch.tensor(0.0, device=z.device)
    n_pairs = 0

    for i in range(n_batches):
        mask_i = batch_labels == unique_batches[i]
        z_i = z[mask_i]
        if z_i.shape[0] < 2:
            continue

        for j in range(i + 1, n_batches):
            mask_j = batch_labels == unique_batches[j]
            z_j = z[mask_j]
            if z_j.shape[0] < 2:
                continue

            total_mmd = total_mmd + mmd_rbf(z_i, z_j, gammas)
            n_pairs += 1

    if n_pairs == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    return total_mmd / n_pairs
