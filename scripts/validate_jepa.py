#!/usr/bin/env python
"""Validate JEPA by extracting embeddings and testing downstream task.

Protocol aligned with transactions_gen_models:
- Global: config/validation/global_target.yaml (LightGBM on train+val bootstrap, test only).
- Local:  config/validation/local_target.yaml (frozen backbone + linear head, last-token local_target).
"""

import argparse
import json
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.jepa import JEPA
from data_utils import (
    get_dataloaders,
    get_preprocessed_splits,
    LocalValidationDataset,
    collate_local_batch,
    collate_mcc_batch,
)


def safe_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC when both classes are present; otherwise NaN (and no sklearn warning)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def extract_embeddings(model, dataloader, device):
    """Extract embeddings from all samples."""
    model.eval()
    
    all_embeddings = []
    all_targets = []
    
    print("Extracting embeddings...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            mcc = batch['mcc'].to(device)
            amount = batch['amount'].to(device)
            time_bucket = batch['time_bucket'].to(device)
            intra_day_rank = batch['intra_day_rank'].to(device)
            target = batch['target'].cpu().numpy()
            
            # Get embeddings (no masking)
            embeddings = model.get_embedding(
                mcc,
                amount,
                time_bucket=time_bucket,
                intra_day_rank=intra_day_rank,
            )  # (batch, seq_len, emb_dim)
            
            # Global pooling (mean over sequence)
            embeddings_global = embeddings.mean(dim=1)  # (batch, emb_dim)
            
            all_embeddings.append(embeddings_global.cpu().numpy())
            all_targets.append(target)
    
    embeddings = np.concatenate(all_embeddings, axis=0)  # (total_samples, emb_dim)
    targets = np.concatenate(all_targets, axis=0)  # (total_samples,)
    
    return embeddings, targets


def train_downstream_classifier(
    train_embeddings,
    train_targets,
    val_embeddings,
    val_targets,
    random_state: int = 42,
    bootstrap: bool = True,
):
    """Train LightGBM on train+val (with optional bootstrap). No val metrics—evaluate on test only.

    Matches transactions_gen_models global_target validation:
    - config/validation/global_target.yaml (LGBM params, random_state=42)
    - embed train+val, bootstrap sample, fit classifier, report test metrics only.
    """
    # Concatenate train+val (reference: global_validation_pipeline uses train+val for fitting)
    train_val_emb = np.concatenate([train_embeddings, val_embeddings], axis=0)
    train_val_tgt = np.concatenate([train_targets, val_targets], axis=0)
    N = len(train_val_emb)

    if bootstrap:
        rng = np.random.default_rng(random_state)
        bootstrap_inds = rng.choice(N, size=N, replace=True)
        fit_emb = train_val_emb[bootstrap_inds]
        fit_tgt = train_val_tgt[bootstrap_inds]
        print("\nTraining downstream classifier (LightGBM on bootstrap sample of train+val)...")
    else:
        fit_emb = train_val_emb
        fit_tgt = train_val_tgt
        print("\nTraining downstream classifier (LightGBM on train+val)...")

    clf = lgb.LGBMClassifier(
        n_estimators=500,
        boosting_type="gbdt",
        subsample=0.5,
        subsample_freq=1,
        learning_rate=0.02,
        feature_fraction=0.75,
        max_depth=6,
        lambda_l1=1,
        lambda_l2=1,
        min_data_in_leaf=50,
        random_state=random_state,
        n_jobs=8,
        verbose=-1,
    )
    clf.fit(fit_emb, fit_tgt)
    return clf


def evaluate_on_test(clf, test_embeddings, test_targets):
    """Evaluate on test set."""
    print("\nEvaluating on test set...")
    
    test_pred = clf.predict(test_embeddings)
    test_pred_proba = clf.predict_proba(test_embeddings)[:, 1]
    
    test_accuracy = accuracy_score(test_targets, test_pred)
    test_auc = safe_roc_auc_score(test_targets, test_pred_proba)
    
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test ROC-AUC: {test_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(test_targets, test_pred, target_names=['Non-Churn', 'Churn']))
    
    return test_accuracy, test_auc


def run_local_validation(
    model: nn.Module,
    parquet_path: Path,
    dataset: str,
    d_model: int,
    device: torch.device,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
    max_epochs: int = 10,
    learning_rate: float = 0.001,
    batch_size: int = 512,
) -> tuple[float, float]:
    """Run local (last-token) validation: frozen backbone + linear head predicting local_target.

    Matches transactions_gen_models config/validation/local_target.yaml:
    - min_len=20, random_min_seq_len=20, random_max_seq_len=40 (train)
    - window_size=32, window_step=16 (val/test)
    - Binary head, BCE, Adam lr=0.001, max_epochs=10, batch_size=512.
    Returns (val_auc, test_auc).
    """
    train_rec, val_rec, test_rec = get_preprocessed_splits(
        parquet_path, dataset=dataset, val_size=val_size, test_size=test_size, random_state=random_state
    )
    train_ds = LocalValidationDataset(
        train_rec,
        min_len=20,
        random_min_seq_len=20,
        random_max_seq_len=40,
        window_size=32,
        window_step=16,
        deterministic=False,
        max_seq_len=40,
        random_state=random_state,
    )
    val_ds = LocalValidationDataset(
        val_rec,
        min_len=20,
        random_min_seq_len=20,
        random_max_seq_len=40,
        window_size=32,
        window_step=16,
        deterministic=True,
        max_seq_len=40,
    )
    test_ds = LocalValidationDataset(
        test_rec,
        min_len=20,
        random_min_seq_len=20,
        random_max_seq_len=40,
        window_size=32,
        window_step=16,
        deterministic=True,
        max_seq_len=40,
    )
    if len(train_ds) == 0:
        print("  No samples for local validation (e.g. no local_target or all seq < 20).")
        return float("nan"), float("nan")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_local_batch, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_local_batch, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_local_batch, num_workers=0
    )
    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    head = nn.Linear(d_model, 1).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    best_val_auc = 0.0
    best_head_state = None
    print("\nLocal validation (last-token local_target)...")
    for epoch in range(max_epochs):
        head.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            mcc = batch["mcc"].to(device)
            amount = batch["amount"].to(device)
            time_bucket = batch["time_bucket"].to(device)
            intra_day_rank = batch["intra_day_rank"].to(device)
            target = batch["local_target"].to(device)
            with torch.no_grad():
                emb = model.get_embedding(
                    mcc, amount, time_bucket=time_bucket, intra_day_rank=intra_day_rank
                )
                emb = emb[:, -1, :]  # last-token, matches last-hidden-state of RNN baselines
            logits = head(emb).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        epoch_loss = epoch_loss / num_batches if num_batches else 0.0
        # Eval on val
        head.eval()
        val_preds, val_tgts = [], []
        with torch.no_grad():
            for batch in val_loader:
                mcc = batch["mcc"].to(device)
                amount = batch["amount"].to(device)
                time_bucket = batch["time_bucket"].to(device)
                intra_day_rank = batch["intra_day_rank"].to(device)
                emb = model.get_embedding(
                    mcc, amount, time_bucket=time_bucket, intra_day_rank=intra_day_rank
                )
                emb = emb[:, -1, :]
                pred = torch.sigmoid(head(emb).squeeze(-1)).cpu().numpy()
                val_preds.append(pred)
                val_tgts.append(batch["local_target"].numpy())
        val_preds = np.concatenate(val_preds, axis=0)
        val_tgts = np.concatenate(val_tgts, axis=0)
        val_auc = safe_roc_auc_score(val_tgts, val_preds)
        if wandb.run is not None:
            wandb.log(
                {"local_train/loss": epoch_loss, "local_val/roc_auc": val_auc if not np.isnan(val_auc) else 0.0},
                step=epoch,
            )
        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    if best_head_state is not None:
        head.load_state_dict(best_head_state)
    head.eval()
    test_preds, test_tgts = [], []
    with torch.no_grad():
        for batch in test_loader:
            mcc = batch["mcc"].to(device)
            amount = batch["amount"].to(device)
            time_bucket = batch["time_bucket"].to(device)
            intra_day_rank = batch["intra_day_rank"].to(device)
            emb = model.get_embedding(
                mcc, amount, time_bucket=time_bucket, intra_day_rank=intra_day_rank
            )
            emb = emb[:, -1, :]
            pred = torch.sigmoid(head(emb).squeeze(-1)).cpu().numpy()
            test_preds.append(pred)
            test_tgts.append(batch["local_target"].numpy())
    test_preds = np.concatenate(test_preds, axis=0)
    test_tgts = np.concatenate(test_tgts, axis=0)
    test_auc = safe_roc_auc_score(test_tgts, test_preds)
    test_acc = accuracy_score(test_tgts, (test_preds >= 0.5).astype(np.float32))
    n_pos = int(test_tgts.sum())
    n_neg = len(test_tgts) - n_pos
    print(f"  Val ROC-AUC:  {best_val_auc:.4f}")
    if np.isnan(test_auc):
        print(f"  Test ROC-AUC: nan (test set has only one class: {n_neg} neg, {n_pos} pos)")
    else:
        print(f"  Test ROC-AUC: {test_auc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    return best_val_auc, test_auc


def run_local_validation_mcc(
    model: nn.Module,
    parquet_path: Path,
    dataset: str,
    d_model: int,
    mcc_vocab_size: int,
    device: torch.device,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
    max_epochs: int = 10,
    learning_rate: float = 0.001,
    batch_size: int = 512,
) -> tuple[float, float]:
    """Run local (last-token) MCC prediction validation: frozen backbone + linear head predicting last mcc_code.

    Matches transactions_gen_models config/validation/event_type.yaml:
    - target_seq_col=mcc_code, same windowing as local_target.
    - Categorical head (num_classes=mcc_vocab_size), CrossEntropyLoss(ignore_index=0), Adam lr=0.001.
    Runs for all datasets (including age). Returns (val_accuracy, test_accuracy).
    """
    train_rec, val_rec, test_rec = get_preprocessed_splits(
        parquet_path, dataset=dataset, val_size=val_size, test_size=test_size, random_state=random_state
    )
    common_kw = dict(
        min_len=20,
        random_min_seq_len=20,
        random_max_seq_len=40,
        window_size=32,
        window_step=16,
        max_seq_len=40,
        target_seq_col="mcc_code",
    )
    train_ds = LocalValidationDataset(
        train_rec, **common_kw, deterministic=False, random_state=random_state
    )
    val_ds = LocalValidationDataset(val_rec, **common_kw, deterministic=True)
    test_ds = LocalValidationDataset(test_rec, **common_kw, deterministic=True)
    if len(train_ds) == 0:
        print("  No samples for local MCC validation.")
        return float("nan"), float("nan")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_mcc_batch, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_mcc_batch, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_mcc_batch, num_workers=0
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    head = nn.Linear(d_model, mcc_vocab_size).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_val_acc = 0.0
    best_head_state = None
    print("\nLocal validation (last-token mcc_code)...")
    for epoch in range(max_epochs):
        head.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            mcc = batch["mcc"].to(device)
            amount = batch["amount"].to(device)
            time_bucket = batch["time_bucket"].to(device)
            intra_day_rank = batch["intra_day_rank"].to(device)
            target = batch["mcc_target"].to(device)
            with torch.no_grad():
                emb = model.get_embedding(
                    mcc, amount, time_bucket=time_bucket, intra_day_rank=intra_day_rank
                )
                emb = emb[:, -1, :]
            logits = head(emb)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        epoch_loss = epoch_loss / num_batches if num_batches else 0.0
        head.eval()
        val_preds, val_tgts = [], []
        with torch.no_grad():
            for batch in val_loader:
                mcc = batch["mcc"].to(device)
                amount = batch["amount"].to(device)
                time_bucket = batch["time_bucket"].to(device)
                intra_day_rank = batch["intra_day_rank"].to(device)
                emb = model.get_embedding(
                    mcc, amount, time_bucket=time_bucket, intra_day_rank=intra_day_rank
                )
                emb = emb[:, -1, :]
                pred = head(emb).argmax(dim=1).cpu().numpy()
                val_preds.append(pred)
                val_tgts.append(batch["mcc_target"].numpy())
        val_preds = np.concatenate(val_preds, axis=0)
        val_tgts = np.concatenate(val_tgts, axis=0)
        # Accuracy ignoring padding (class 0)
        val_mask = val_tgts != 0
        val_acc = (val_preds[val_mask] == val_tgts[val_mask]).mean() if val_mask.any() else 0.0
        if wandb.run is not None:
            wandb.log(
                {"local_mcc_train/loss": epoch_loss, "local_mcc_val/accuracy": val_acc},
                step=epoch,
            )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    if best_head_state is not None:
        head.load_state_dict(best_head_state)
    head.eval()
    test_preds, test_tgts = [], []
    with torch.no_grad():
        for batch in test_loader:
            mcc = batch["mcc"].to(device)
            amount = batch["amount"].to(device)
            time_bucket = batch["time_bucket"].to(device)
            intra_day_rank = batch["intra_day_rank"].to(device)
            emb = model.get_embedding(
                mcc, amount, time_bucket=time_bucket, intra_day_rank=intra_day_rank
            )
            emb = emb[:, -1, :]
            pred = head(emb).argmax(dim=1).cpu().numpy()
            test_preds.append(pred)
            test_tgts.append(batch["mcc_target"].numpy())
    test_preds = np.concatenate(test_preds, axis=0)
    test_tgts = np.concatenate(test_tgts, axis=0)
    test_mask = test_tgts != 0
    test_acc = (test_preds[test_mask] == test_tgts[test_mask]).mean() if test_mask.any() else 0.0
    print(f"  Val Accuracy:  {best_val_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    return best_val_acc, test_acc


def parse_args():
    parser = argparse.ArgumentParser(description="Validate JEPA on downstream task.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Experiment folder: name (e.g. jepa_20250101_120000) under outputs/, or path to folder or best_model.pt. Default: latest jepa_* in outputs/.",
    )
    return parser.parse_args()


def main(model_path: str = None):
    """Main validation script."""
    args = parse_args()
    # CLI overrides: --model-dir takes precedence over model_path (for script callers)
    model_dir_arg = args.model_dir if args.model_dir is not None else model_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    outputs_dir = Path('outputs')

    # Resolve experiment directory and model path
    if model_dir_arg is None:
        model_dirs = sorted(outputs_dir.glob('jepa_*'))
        if not model_dirs:
            print("✗ No trained models found in outputs/")
            return
        exp_dir = model_dirs[-1]
        model_path = exp_dir / 'best_model.pt'
        print(f"Using model: {model_path}\n")
    else:
        p = Path(model_dir_arg)
        if p.is_file():
            exp_dir = p.parent
            model_path = p
        elif p.is_dir():
            exp_dir = p
            model_path = exp_dir / 'best_model.pt'
        else:
            # Treat as folder name under outputs/
            exp_dir = outputs_dir / p
            model_path = exp_dir / 'best_model.pt'
        if not exp_dir.exists():
            print(f"✗ Not found: {exp_dir}")
            return
        print(f"Using model: {model_path}\n")

    if not model_path.exists():
        print(f"✗ Weights not found: {model_path}")
        return
    config_path = exp_dir / 'config.json'
    if not config_path.exists():
        print(f"✗ Config not found: {config_path}")
        print("  Validation requires the experiment config (saved by train_jepa.py).")
        return
    with open(config_path) as f:
        config = json.load(f)

    wandb_config = {**config, "model_path": str(model_path), "exp_dir": str(exp_dir)}
    wandb.init(project="latentledger", job_type="validation", config=wandb_config)

    # Build model from saved config
    print("Loading JEPA model...")
    model = JEPA(
        mcc_vocab_size=config['mcc_vocab_size'],
        mcc_emb_dim=config['mcc_emb_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        predictor_d_model=config['predictor_d_model'],
        dropout=config['dropout'],
        use_temporal_encoding=config.get('use_temporal_encoding', False),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"  Model loaded from: {model_path}")
    print(f"  Config from: {config_path}\n")

    # Load data: same parquet/dataset/split as training (transactions_gen_models protocol)
    parquet_path = Path(config['parquet_path'])
    dataset = config.get('dataset', 'churn')
    max_seq_len = config.get('max_seq_len', 256)
    val_size = config.get('val_size', 0.1)
    test_size = config.get('test_size', 0.1)
    random_state = config.get('random_state', 42)
    # Embedding extraction uses batch_size=64 to match config/validation/global_target.yaml embed_data
    val_batch_size = 64
    train_loader, val_loader, test_loader = get_dataloaders(
        parquet_path,
        dataset=dataset,
        batch_size=val_batch_size,
        max_seq_len=max_seq_len,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )
    
    # Extract embeddings
    train_emb, train_targets = extract_embeddings(model, train_loader, device)
    val_emb, val_targets = extract_embeddings(model, val_loader, device)
    test_emb, test_targets = extract_embeddings(model, test_loader, device)
    
    print("\nEmbedding shapes:")
    print(f"  Train: {train_emb.shape}")
    print(f"  Val: {val_emb.shape}")
    print(f"  Test: {test_emb.shape}")
    
    # Train downstream classifier on train+val (bootstrap), evaluate on test only (transactions_gen_models protocol)
    clf = train_downstream_classifier(
        train_emb,
        train_targets,
        val_emb,
        val_targets,
        random_state=random_state,
        bootstrap=True,
    )
    
    # Evaluate on test only (reference reports test metrics only)
    test_acc, test_auc = evaluate_on_test(clf, test_emb, test_targets)
    wandb.log({"global/test_accuracy": test_acc, "global/test_roc_auc": test_auc})

    # Print summary (protocol: transactions_gen_models global_target)
    print("\n" + "=" * 60)
    print("GLOBAL VALIDATION (global_target protocol)")
    print("=" * 60)
    print(f"Test ROC-AUC:  {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("=" * 60)

    # Local validation: label prediction (skip for age: no local_target)
    local_val_auc, local_test_auc = float("nan"), float("nan")
    if dataset not in ("age", "age_nodup"):
        local_val_auc, local_test_auc = run_local_validation(
            model,
            parquet_path,
            dataset=dataset,
            d_model=config["d_model"],
            device=device,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            max_epochs=10,
            learning_rate=0.001,
            batch_size=512,
        )
        wandb.log({"local/val_roc_auc": local_val_auc, "local/test_roc_auc": local_test_auc})
        print("\n" + "=" * 60)
        print("LOCAL VALIDATION (local_target protocol)")
        print("=" * 60)
        print(f"Val ROC-AUC:  {local_val_auc:.4f}")
        print(f"Test ROC-AUC: {local_test_auc:.4f}")
        print("=" * 60)
    else:
        print("\n(Skipping local_target validation: age dataset has no local_target)")

    # Local validation: MCC code prediction (all datasets, including age)
    mcc_vocab_size = config.get("mcc_vocab_size", 101)
    local_mcc_val_acc, local_mcc_test_acc = run_local_validation_mcc(
        model,
        parquet_path,
        dataset=dataset,
        d_model=config["d_model"],
        mcc_vocab_size=mcc_vocab_size,
        device=device,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        max_epochs=10,
        learning_rate=0.001,
        batch_size=512,
    )
    wandb.log({"local_mcc/val_accuracy": local_mcc_val_acc, "local_mcc/test_accuracy": local_mcc_test_acc})
    print("\n" + "=" * 60)
    print("LOCAL VALIDATION (mcc_code / event_type protocol)")
    print("=" * 60)
    print(f"Val Accuracy:  {local_mcc_val_acc:.4f}")
    print(f"Test Accuracy: {local_mcc_test_acc:.4f}")
    print("=" * 60)

    wandb.finish()
    # Success criteria (global test AUC)
    if test_auc > 0.65:
        print("\n✓ JEPA validation PASSED!")
        print("  (Global test ROC-AUC > 0.65)")
        return True
    else:
        print("\n⚠ JEPA validation MARGINAL")
        print(f"  (Global test ROC-AUC = {test_auc:.4f}, target > 0.65)")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)