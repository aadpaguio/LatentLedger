"""Collapse and representation diagnostics for JEPA training."""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _effective_rank(H: np.ndarray) -> float:
    """Effective rank from eigenvalue entropy of covariance matrix.

    H: (B, D) centered embeddings.
    Returns exp(entropy(normalized_eigenvalues)).
    """
    C = (H.T @ H) / H.shape[0]  # (D, D)
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = np.maximum(eigenvalues, 0)  # numerical safety
    total = eigenvalues.sum()
    if total < 1e-12:
        return 0.0
    p = eigenvalues / total
    p = p[p > 1e-12]  # drop zeros before log
    entropy = -(p * np.log(p)).sum()
    return float(np.exp(entropy))


def compute_collapse_diagnostics(
    model,
    val_loader,
    device: torch.device,
    n_batches: int = 3,
) -> dict:
    """Collapse diagnostics for JEPA training.

    Runs on n_batches from val_loader. Returns flat dict for wandb.log().

    Metrics:
      diagnostics/effective_rank      - exp(entropy) of covariance spectrum (target encoder)
      diagnostics/mean_dim_variance   - mean per-dimension variance (target encoder)
      diagnostics/dead_dims           - dims with variance < 1e-6 (target encoder)
      diagnostics/param_l2_distance   - L2 distance between context/target encoder params
      diagnostics/encoder_cosine_sim  - mean cosine sim between context/target mean-pooled embeddings
      diagnostics/probe_roc_auc       - logistic regression on context encoder embeddings vs target label
    """
    model.eval()

    # -- Collect embeddings from both encoders --
    ctx_embs = []
    tgt_embs = []
    labels = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            mcc = batch['mcc'].to(device)
            amount = batch['amount'].to(device)
            time_bucket = batch['time_bucket'].to(device)
            intra_day_rank = batch['intra_day_rank'].to(device)

            # Context encoder (via get_embedding — matches validate_jepa.py)
            ctx = model.get_embedding(
                mcc, amount,
                time_bucket=time_bucket,
                intra_day_rank=intra_day_rank,
            )  # (B, T, D)
            ctx_pooled = ctx.mean(dim=1)  # (B, D)

            # Target encoder
            tgt = model.target_encoder(
                mcc, amount,
                time_bucket=time_bucket,
                intra_day_rank=intra_day_rank,
            )  # (B, T, D)
            tgt_pooled = tgt.mean(dim=1)  # (B, D)

            ctx_embs.append(ctx_pooled.cpu().numpy())
            tgt_embs.append(tgt_pooled.cpu().numpy())

            if 'target' in batch:
                labels.append(batch['target'].numpy())

    ctx_embs = np.concatenate(ctx_embs, axis=0)   # (N, D)
    tgt_embs = np.concatenate(tgt_embs, axis=0)   # (N, D)
    labels = np.concatenate(labels, axis=0) if labels else None

    metrics = {}

    # -- 1. Effective rank (target encoder) --
    H = tgt_embs - tgt_embs.mean(axis=0, keepdims=True)  # center
    metrics['diagnostics/effective_rank'] = _effective_rank(H)

    # -- 2. Variance stats (target encoder) --
    dim_var = tgt_embs.var(axis=0)  # (D,)
    metrics['diagnostics/mean_dim_variance'] = float(dim_var.mean())
    metrics['diagnostics/dead_dims'] = int((dim_var < 1e-6).sum())

    # -- 3. Parameter L2 distance --
    l2_sq = 0.0
    for p_ctx, p_tgt in zip(
        model.context_encoder.parameters(),
        model.target_encoder.parameters(),
    ):
        l2_sq += (p_ctx.data - p_tgt.data).pow(2).sum().item()
    metrics['diagnostics/param_l2_distance'] = float(np.sqrt(l2_sq))

    # -- 4. Encoder cosine similarity --
    # Row-wise cosine between context and target mean-pooled embeddings
    ctx_t = torch.from_numpy(ctx_embs)
    tgt_t = torch.from_numpy(tgt_embs)
    cos_sim = F.cosine_similarity(ctx_t, tgt_t, dim=1)  # (N,)
    metrics['diagnostics/encoder_cosine_sim'] = float(cos_sim.mean().item())

    # -- 5. Linear probe (if labels available and both classes present) --
    if labels is not None and len(np.unique(labels)) >= 2:
        N = len(ctx_embs)
        # 80/20 split (deterministic)
        split = int(0.8 * N)
        try:
            clf = LogisticRegression(max_iter=200, solver='lbfgs')
            clf.fit(ctx_embs[:split], labels[:split])
            probe_proba = clf.predict_proba(ctx_embs[split:])[:, 1]
            probe_auc = roc_auc_score(labels[split:], probe_proba)
            metrics['diagnostics/probe_roc_auc'] = float(probe_auc)
        except Exception:
            metrics['diagnostics/probe_roc_auc'] = float('nan')
    else:
        metrics['diagnostics/probe_roc_auc'] = float('nan')

    return metrics
