#!/usr/bin/env python
"""Shared JEPA training loop (Hydra DictConfig or plain dict via OmegaConf)."""

from __future__ import annotations

import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np
import torch
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_utils import get_dataloaders
from diagnostics import compute_collapse_diagnostics
from models.jepa import JEPA

logger = logging.getLogger(__name__)

ConfigLike = Union[DictConfig, Mapping[str, Any]]


def load_composed_config(
    dataset: str = "churn",
    model: str = "small",
    training_name: str = "default",
) -> DictConfig:
    """Compose the same structure as Hydra defaults without Hydra (for legacy train_jepa.py)."""
    root = Path(__file__).resolve().parent.parent
    meta = OmegaConf.load(root / "configs" / "config.yaml")
    ds_path = root / "configs" / "dataset" / f"{dataset}.yaml"
    if not ds_path.exists():
        # e.g. churn_nodup shares churn.yaml
        stem = dataset.replace("_nodup", "")
        ds_path = root / "configs" / "dataset" / f"{stem}.yaml"
    ds = OmegaConf.load(ds_path)
    if str(ds.get("name", "")) != dataset:
        ds.name = dataset
    mo = OmegaConf.load(root / "configs" / "model" / f"{model}.yaml")
    tr = OmegaConf.load(root / "configs" / "training" / f"{training_name}.yaml")
    cfg = OmegaConf.create(
        {
            "seed": meta.seed,
            "use_temporal_encoding": meta.use_temporal_encoding,
            "device": meta.device,
            "wandb": meta.wandb,
            "dataset": ds,
            "model": mo,
            "training": tr,
        }
    )
    OmegaConf.resolve(cfg)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: ConfigLike) -> torch.device:
    d = OmegaConf.select(cfg, "device", default=None) if isinstance(cfg, DictConfig) else cfg.get("device")
    if d is not None and str(d).lower() not in ("null", "none", ""):
        return torch.device(str(d))
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_container(cfg: ConfigLike) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def apply_temporal_fallback(cfg: DictConfig) -> None:
    """Mutates cfg: disable temporal encoding when dataset has no timestamps."""
    if not cfg.dataset.get("has_timestamps", True) and cfg.get("use_temporal_encoding", False):
        logger.warning(
            "use_temporal_encoding=true but dataset.has_timestamps=false; using ordinal encoding."
        )
        cfg.use_temporal_encoding = False


def flatten_config_for_json(cfg: DictConfig) -> dict[str, Any]:
    """Flat dict for backward compatibility with validate_jepa.py (reads config.json)."""
    c = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(c, dict)
    ds = c.get("dataset") or {}
    tr = c.get("training") or {}
    mo = c.get("model") or {}
    wb = c.get("wandb") or {}
    flat: dict[str, Any] = {
        "seed": c.get("seed", 42),
        "use_temporal_encoding": c.get("use_temporal_encoding", False),
        "parquet_path": ds.get("parquet_path"),
        "dataset": ds.get("name"),
        "max_seq_len": ds.get("max_seq_len", 256),
        "mcc_vocab_size": ds.get("mcc_vocab_size", 101),
        "mcc_emb_dim": ds.get("mcc_emb_dim", 24),
        "d_model": mo.get("d_model"),
        "nhead": mo.get("nhead"),
        "num_layers": mo.get("num_layers"),
        "predictor_d_model": mo.get("predictor_d_model"),
        "predictor_num_layers": mo.get("predictor_num_layers", 12),
        "dropout": mo.get("dropout", 0.1),
        "learning_rate": tr.get("learning_rate"),
        "weight_decay": tr.get("weight_decay", 0.4),
        "use_cosine_annealing": tr.get("use_cosine_annealing", True),
        "epochs": tr.get("epochs", 60),
        "batch_size": tr.get("batch_size", 128),
        "ema_tau_start": tr.get("ema_tau_start", 0.90),
        "ema_tau_end": tr.get("ema_tau_end", 0.999),
        "grad_clip_max_norm": tr.get("grad_clip_max_norm", 1.0),
        "val_size": tr.get("val_size", 0.1),
        "test_size": tr.get("test_size", 0.1),
        "random_state": tr.get("random_state", 42),
        "num_workers": tr.get("num_workers", 0),
        "wandb_project": wb.get("project", "latentledger"),
        "wandb_group": wb.get("group", "benchmark_v1"),
    }
    dev = c.get("device")
    if dev is not None:
        flat["device"] = dev
    return flat


def train_epoch(
    model: JEPA,
    train_loader,
    optimizer,
    device: torch.device,
    epoch: int,
    total_steps: int,
    steps_completed: int,
    grad_clip_max_norm: float,
    ema_tau_start: float,
    ema_tau_end: float,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")

    for batch_idx, batch in enumerate(pbar):
        mcc = batch["mcc"].to(device)
        amount = batch["amount"].to(device)
        time_bucket = batch["time_bucket"].to(device)
        intra_day_rank = batch["intra_day_rank"].to(device)

        loss, _ = model(
            mcc,
            amount,
            time_bucket=time_bucket,
            intra_day_rank=intra_day_rank,
        )

        optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()

        step = steps_completed + batch_idx
        tau = model.get_ema_decay(
            step, total_steps, tau_start=ema_tau_start, tau_end=ema_tau_end
        )
        model._update_target_encoder(tau=tau)

        wandb.log({"train/step_loss": loss.item(), "train/grad_norm": total_norm}, step=step)
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / num_batches


def validate(model, val_loader, device: torch.device, debug: bool = True) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            mcc = batch["mcc"].to(device)
            amount = batch["amount"].to(device)
            time_bucket = batch["time_bucket"].to(device)
            intra_day_rank = batch["intra_day_rank"].to(device)

            loss, sx = model(
                mcc,
                amount,
                time_bucket=time_bucket,
                intra_day_rank=intra_day_rank,
            )

            total_loss += loss.item()
            num_batches += 1

            if debug and batch_idx == 0:
                sy = model.target_encoder(
                    mcc,
                    amount,
                    time_bucket=time_bucket,
                    intra_day_rank=intra_day_rank,
                )
                print("\n  [Debug - first val batch]")
                print(f"  Target encoder output norm:   {sy.norm(dim=-1).mean():.4f}")
                print(f"  Context encoder output norm:  {sx.norm(dim=-1).mean():.4f}")
                print(f"  Target std across batch:      {sy.std(dim=0).mean():.6f}")
                print(f"  Target mean across batch:     {sy.mean():.6f}")
                print(f"  Loss this batch:              {loss.item():.6f}")

    return total_loss / num_batches


def get_project_root() -> Path:
    """Hydra cwd when launched with @hydra.main; else repo root (e.g. train_jepa.py)."""
    try:
        import hydra

        return Path(hydra.utils.get_original_cwd())
    except (ImportError, ValueError):
        return Path(__file__).resolve().parent.parent


def run_training(cfg: DictConfig, hydra_output_dir: Path | None = None) -> Path:
    """Train JEPA; saves flat config.json for validate_jepa compatibility."""
    apply_temporal_fallback(cfg)
    set_seed(int(cfg.seed))

    project_root = get_project_root()

    if hydra_output_dir is not None:
        exp_dir = Path(hydra_output_dir)
    else:
        output_dir = project_root / "outputs"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = output_dir / f"jepa_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    flat = flatten_config_for_json(cfg)
    parquet_rel = flat["parquet_path"]
    parquet_path = (project_root / parquet_rel).resolve() if not Path(parquet_rel).is_absolute() else Path(parquet_rel)
    flat["parquet_path"] = str(parquet_path)

    device = resolve_device(cfg)
    flat["device"] = str(device)
    print(f"Using device: {device}")

    with open(exp_dir / "config.json", "w") as f:
        json.dump(flat, f, indent=2)

    enc_mode = "temporal" if cfg.use_temporal_encoding else "ordinal"
    run_name = f"{cfg.dataset.name}_{cfg.model.size_name}_{enc_mode}_{cfg.seed}"
    tags = [
        "jepa",
        str(cfg.dataset.name),
        str(cfg.model.size_name),
        enc_mode,
    ]
    wandb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        name=run_name,
        tags=tags,
        job_type="training",
        config=_to_container(cfg),
    )

    model = JEPA(
        mcc_vocab_size=int(cfg.dataset.mcc_vocab_size),
        mcc_emb_dim=int(cfg.dataset.mcc_emb_dim),
        d_model=int(cfg.model.d_model),
        nhead=int(cfg.model.nhead),
        num_layers=int(cfg.model.num_layers),
        predictor_d_model=int(cfg.model.predictor_d_model),
        predictor_num_layers=int(cfg.model.predictor_num_layers),
        dropout=float(cfg.model.dropout),
        use_temporal_encoding=bool(cfg.use_temporal_encoding),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )

    scheduler = None
    if bool(cfg.training.use_cosine_annealing):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg.training.epochs), eta_min=1e-6
        )

    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        parquet_path,
        dataset=str(cfg.dataset.name),
        batch_size=int(cfg.training.batch_size),
        max_seq_len=int(cfg.dataset.max_seq_len),
        num_workers=int(cfg.training.num_workers),
        val_size=float(cfg.training.val_size),
        test_size=float(cfg.training.test_size),
        random_state=int(cfg.training.random_state),
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    total_steps = len(train_loader) * int(cfg.training.epochs)
    print("\nStarting training...\n")

    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0
    steps_completed = 0
    epoch = 0
    ema_start = float(cfg.training.ema_tau_start)
    ema_end = float(cfg.training.ema_tau_end)
    grad_clip = float(cfg.training.grad_clip_max_norm)

    for epoch in range(1, int(cfg.training.epochs) + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            total_steps,
            steps_completed,
            grad_clip,
            ema_start,
            ema_end,
        )
        steps_completed += len(train_loader)

        val_loss = validate(model, val_loader, device)

        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else float(cfg.training.learning_rate)
        wandb.log(
            {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr": current_lr,
            },
            step=steps_completed,
        )
        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch}/{cfg.training.epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), exp_dir / "best_model.pt")
            print("  → Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch} epochs")
                break

        diag = compute_collapse_diagnostics(model, val_loader, device, n_batches=3)
        wandb.log(diag, step=steps_completed)
        dead_dims = diag["diagnostics/dead_dims"]
        d_model = int(model.d_model)
        token_dead_dims = diag["diagnostics/token_dead_dims"]
        print(
            f"  eff_rank={diag['diagnostics/effective_rank']:.1f}  "
            f"cos_sim={diag['diagnostics/encoder_cosine_sim']:.4f}  "
            f"probe_auc={diag['diagnostics/probe_roc_auc']:.3f}  "
            f"param_l2={diag['diagnostics/param_l2_distance']:.2f}  "
            f"dead_dims(pooled)={dead_dims}/{d_model}  "
            f"dead_dims(token)={token_dead_dims}/{d_model}"
        )
        if dead_dims > d_model * 0.5:
            print(
                f"\nEarly stopping: dead_dims ({dead_dims}) > 0.5 * d_model ({d_model * 0.5:.0f})"
            )
            wandb.log(
                {"train/early_stop_reason": "dead_dims_threshold"},
                step=steps_completed,
            )
            break

    wandb.finish()

    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(exp_dir / "best_model.pt", map_location=device))
    test_loss = validate(model, test_loader, device, debug=False)
    print(f"Test Loss: {test_loss:.4f}")

    results = {
        "config": flat,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "epochs_trained": epoch,
    }
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ Training complete!")
    print(f"  Outputs saved to: {exp_dir}")
    return exp_dir
