#!/usr/bin/env python
"""JEPA: Joint Embedding Predictive Architecture for Transaction Sequences."""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

# Safe default for positional embedding; sequences longer than this are clamped.
MAX_SEQ_LEN = 512
# Number of target blocks for I-JEPA multi-block masking (Section 3, Appendix A.1).
NUM_TARGET_BLOCKS = 4


class TransactionEncoder(nn.Module):
    """Transformer encoder for transaction sequences (matches MLM baseline in paper Appendix A)."""

    def __init__(
        self,
        mcc_vocab_size: int,
        mcc_emb_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.mcc_emb_dim = mcc_emb_dim

        # Paper clips rare MCCs to top-100 per dataset; mcc_vocab_size should be set to 101 (0-indexed + 1 for mask token)
        self.mcc_embedding = nn.Embedding(mcc_vocab_size, mcc_emb_dim)

        # Project [mcc_emb (mcc_emb_dim) + amount (1)] to d_model
        self.input_proj = nn.Linear(mcc_emb_dim + 1, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Learned positional encodings
        self.pos_embedding = nn.Embedding(MAX_SEQ_LEN, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        mcc: torch.Tensor,
        amount: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            mcc: (batch, seq_len) - MCC codes
            amount: (batch, seq_len, 1) - Transaction amounts
            position_ids: Optional (batch, seq_len) - original position indices for pos embedding.
                When provided (e.g. context-only encoding), use these instead of arange(seq_len).

        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        batch_size, seq_len = mcc.shape
        device = mcc.device

        # Embed MCC
        mcc_emb = self.mcc_embedding(mcc)  # (batch, seq_len, mcc_emb_dim)

        # Paper Eq. 5: sign(a) * ln(1 + |a|)
        amount_norm = torch.sign(amount) * torch.log1p(torch.abs(amount))  # (batch, seq_len, 1)

        # Concatenate and project to d_model
        combined = torch.cat([mcc_emb, amount_norm], dim=-1)  # (batch, seq_len, mcc_emb_dim + 1)
        x = self.input_norm(self.input_proj(combined))  # (batch, seq_len, d_model)

        # Positional encoding: use provided indices (e.g. context positions) or 0..seq_len-1
        if position_ids is not None:
            pos_ids = position_ids.clamp(max=MAX_SEQ_LEN - 1)
            x = x + self.pos_embedding(pos_ids)
        else:
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).clamp(max=MAX_SEQ_LEN - 1)
            x = x + self.pos_embedding(position_ids).unsqueeze(0)

        # Transformer encoder
        out = self.transformer(x)  # (batch, seq_len, d_model)
        return out


class Predictor(nn.Module):
    """
    Narrow self-attention ViT predictor (I-JEPA Appendix A.1, Meta VisionTransformerPredictor).

    Context tokens are projected to predictor dim; mask token and pos embeddings live in predictor dim.
    Context + target tokens are concatenated and fed through self-attention (not cross-attention).
    """

    def __init__(
        self,
        encoder_d_model: int,
        predictor_d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(encoder_d_model, predictor_d_model)
        self.predictor_pos_embed = nn.Embedding(MAX_SEQ_LEN, predictor_d_model)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_d_model,
            nhead=nhead,
            dim_feedforward=predictor_d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.predictor_norm = nn.LayerNorm(predictor_d_model)
        self.predictor_proj = nn.Linear(predictor_d_model, encoder_d_model)

    def forward(
        self,
        sx: torch.Tensor,  # (batch, N_ctxt, encoder_d_model) — context encoder output
        context_positions: torch.Tensor,  # (batch, N_ctxt) — original position indices
        target_positions: torch.Tensor,  # (batch, N_tgt) — original position indices
    ) -> torch.Tensor:
        """
        Returns:
            predictions: (batch, N_tgt, encoder_d_model)
        """
        batch_size = sx.size(0)
        N_ctxt = sx.size(1)

        # Clamp position indices for embedding lookup
        context_positions = context_positions.clamp(max=MAX_SEQ_LEN - 1)
        target_positions = target_positions.clamp(max=MAX_SEQ_LEN - 1)

        # Project context to predictor dim and add pos embeddings
        x = self.predictor_embed(sx)
        x = x + self.predictor_pos_embed(context_positions)

        # Build target tokens: mask token + pos embedding (both in predictor dim)
        pred_tokens = self.mask_token.expand(batch_size, target_positions.size(1), -1)
        pred_tokens = pred_tokens + self.predictor_pos_embed(target_positions)

        # Concatenate context + target, run self-attention
        x = torch.cat([x, pred_tokens], dim=1)
        x = self.transformer(x)

        # Extract target outputs only, norm, project back to encoder dim
        x = x[:, N_ctxt:]
        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        return x


class JEPA(nn.Module):
    """
    Joint Embedding Predictive Architecture for Transaction Sequences.

    Key idea: Instead of predicting in input space (like MAE),
    predict in embedding space. Uses EMA target encoder for stability.
    """

    def __init__(
        self,
        mcc_vocab_size: int,  # required. Paper: 101 for Churn/HSBC/Default, 101 for Age (top-100 + mask)
        mcc_emb_dim: int = 24,  # matches paper Appendix A (24 for Churn/HSBC, 16 for Age/Default)
        d_model: int = 1024,  # matches MLM baseline in paper
        nhead: int = 8,  # matches MLM baseline in paper
        num_layers: int = 6,  # matches MLM baseline in paper
        predictor_d_model: int = 384,  # narrow predictor bottleneck (I-JEPA Appendix A.1)
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Online (context) encoder
        self.online_encoder = TransactionEncoder(
            mcc_vocab_size=mcc_vocab_size,
            mcc_emb_dim=mcc_emb_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Target encoder (EMA of online)
        self.target_encoder = TransactionEncoder(
            mcc_vocab_size=mcc_vocab_size,
            mcc_emb_dim=mcc_emb_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Copy weights
        self._update_target_encoder(tau=1.0)

        # Predictor: narrow self-attention ViT (Appendix A.1); mask token and pos embed live inside it
        self.predictor = Predictor(
            encoder_d_model=d_model,
            predictor_d_model=predictor_d_model,
            nhead=nhead,
            num_layers=12,
            dropout=dropout,
        )

        # Don't train target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def get_ema_decay(self, step: int, total_steps: int) -> float:
      return 0.996 + (1.0 - 0.996) * (step / max(1, total_steps))

    def _update_target_encoder(self, tau: float):
        """
        Update target encoder with EMA.
        Call in the training loop after each optimizer step with:
          tau = model.get_ema_decay(step, total_steps)
          model._update_target_encoder(tau=tau)
        """
        for online_param, target_param in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_param.data = tau * target_param.data + (1 - tau) * online_param.data.detach()

    def sample_masks(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        I-JEPA multi-block masking — positions sampled independently per sample.

        Block *lengths* are shared across the batch (same scale drawn once) so tensors
        stay rectangular and need no padding. Block *start positions* are drawn
        independently for every sample in the batch.

        Returns:
            context_indices : (batch_size, num_context_positions)
            target_blocks   : list of M tensors, each (batch_size, block_len_i)
        """
        M = NUM_TARGET_BLOCKS

        # ── 1. Target blocks ──────────────────────────────────────────────────────
        # Draw block lengths once (shared scale → uniform tensor shapes).
        block_lens = [
            max(1, int(random.uniform(0.15 * seq_len, 0.2 * seq_len)))
            for _ in range(M)
        ]

        target_blocks: List[torch.Tensor] = []
        # Boolean occupancy mask: True where ANY target block covers each position.
        target_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        for i in range(M):
            blen = block_lens[i]
            max_start = max(0, seq_len - blen)
            # Independent start per sample: (batch_size,)
            starts = torch.randint(0, max_start + 1, (batch_size,), device=device)
            # Build index tensor: (batch_size, blen)
            offsets = torch.arange(blen, device=device).unsqueeze(0)          # (1, blen)
            indices = (starts.unsqueeze(1) + offsets).clamp(max=seq_len - 1)  # (batch_size, blen)
            target_blocks.append(indices)
            # Mark positions as target
            target_mask.scatter_(1, indices, True)

        # ── 2. Context block ──────────────────────────────────────────────────────
        # Same length drawn once; per-sample start positions.
        context_len = min(int(random.uniform(0.85 * seq_len, seq_len)), seq_len)
        max_start = max(0, seq_len - context_len)
        ctx_starts = torch.randint(0, max_start + 1, (batch_size,), device=device)
        offsets = torch.arange(context_len, device=device).unsqueeze(0)
        ctx_candidates = (ctx_starts.unsqueeze(1) + offsets).clamp(max=seq_len - 1)  # (batch_size, context_len)

        # Per-sample boolean mask of candidate context positions
        ctx_candidate_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        ctx_candidate_mask.scatter_(1, ctx_candidates, True)

        # Remove overlap with target positions
        ctx_valid_mask = ctx_candidate_mask & ~target_mask  # (batch_size, seq_len)

        # Fallback: any sample with zero valid context positions gets all non-target positions
        empty_rows = ctx_valid_mask.sum(dim=1) == 0            # (batch_size,) bool
        if empty_rows.any():
            fallback = ~target_mask                            # non-target positions
            ctx_valid_mask[empty_rows] = fallback[empty_rows]
        # Final fallback (all positions are target): expose position 0 for those rows
        still_empty = ctx_valid_mask.sum(dim=1) == 0
        if still_empty.any():
            ctx_valid_mask[still_empty, 0] = True

        # ── 3. Build rectangular context tensor ───────────────────────────────────
        # Truncate to the *minimum* valid context length across the batch so that the
        # tensor is rectangular. This is conservative but keeps shapes fixed.
        min_ctx = int(ctx_valid_mask.sum(dim=1).min().item())
        min_ctx = max(1, min_ctx)

        # For each sample, collect the first `min_ctx` valid positions (sorted ascending).
        # nonzero on CPU is fastest; for GPU, argsort on the bool mask works.
        context_indices_list = []
        for b in range(batch_size):
            pos = ctx_valid_mask[b].nonzero(as_tuple=False).squeeze(1)  # sorted ascending
            context_indices_list.append(pos[:min_ctx])
        context_indices = torch.stack(context_indices_list, dim=0)  # (batch_size, min_ctx)

        return context_indices, target_blocks

    def forward(
        self,
        mcc: torch.Tensor,
        amount: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mcc: (batch, seq_len) - MCC codes
            amount: (batch, seq_len, 1) - Transaction amounts

        Returns:
            loss: Scalar loss (mean over M blocks)
            sx: (batch, num_context, d_model) - Context encoder output
        """
        batch_size, seq_len = mcc.shape
        device = mcc.device
        d_model = self.d_model
        M = NUM_TARGET_BLOCKS

        context_indices, target_blocks = self.sample_masks(batch_size, seq_len, device)

        # Gather only context positions — context encoder sees no masked positions (Fix 2)
        mcc_context = torch.gather(
            mcc, 1, context_indices
        )  # (batch, num_context)
        amount_context = torch.gather(
            amount, 1, context_indices.unsqueeze(-1).expand(-1, -1, 1)
        )  # (batch, num_context, 1)

        sx = self.online_encoder(
            mcc_context, amount_context, position_ids=context_indices
        )  # (batch, num_context, d_model)

        with torch.no_grad():
            sy = self.target_encoder(mcc, amount)  # (batch, seq_len, d_model)

        total_loss = 0.0
        for i in range(M):
            block_positions = target_blocks[i]  # (batch, block_len) — position indices

            pred_i = self.predictor(sx, context_indices, block_positions)  # (batch, block_len, d_model)
            target_i = sy.gather(
                1, block_positions.unsqueeze(-1).expand(-1, -1, d_model)
            )
            total_loss += F.mse_loss(pred_i, target_i.detach(), reduction="mean")

        loss = total_loss / M
        return loss, sx

    def get_embedding(self, mcc: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
        """
        Get token-level embeddings without masking (for local evaluation, e.g. sliding window Section 3.2).
        Uses target encoder for evaluation (Appendix A.1).
        """
        with torch.no_grad():
            return self.target_encoder(mcc, amount)

    def get_global_embedding(self, mcc: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool token embeddings for global evaluation (fed to LightGBM).
        Uses target encoder for evaluation (Appendix A.1).
        """
        with torch.no_grad():
            token_embeddings = self.target_encoder(mcc, amount)
        return token_embeddings.mean(dim=1)
