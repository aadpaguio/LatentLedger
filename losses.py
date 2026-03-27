"""VICReg auxiliary losses for collapse prevention."""

import torch
import torch.nn.functional as F


def vicreg_var_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Variance loss: penalise dimensions whose std < gamma. z: (N, D)."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(gamma - std).mean()


def vicreg_cov_loss(z: torch.Tensor) -> torch.Tensor:
    """Covariance loss: penalise off-diagonal entries of normalised cov matrix. z: (N, D)."""
    N, D = z.shape
    if N < 2:
        return z.new_zeros(())
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (N - 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / D
