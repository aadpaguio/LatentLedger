"""Loss functions for representation learning (e.g. VICReg-style regularization)."""

import torch
import torch.nn.functional as F


def vicreg_cov_loss(z: torch.Tensor) -> torch.Tensor:
    """Covariance regularization on (B, D) embeddings.

    Penalizes off-diagonal covariance entries → decorrelates dimensions.
    """
    B, D = z.shape
    if B < 2:
        return z.new_zeros(1).squeeze()
    z = z - z.mean(dim=0)          # center
    cov = (z.T @ z) / (B - 1)     # (D, D)
    # Zero out diagonal (we only penalize off-diagonal)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / D


def vicreg_var_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Variance regularization: push per-dim std above gamma."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))
