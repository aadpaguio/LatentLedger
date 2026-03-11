#!/usr/bin/env python
"""Validate JEPA by extracting embeddings and testing downstream task."""

import argparse
import json
import torch
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.jepa import JEPA
from data_utils import get_dataloaders


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


def train_downstream_classifier(train_embeddings, train_targets, val_embeddings, val_targets):
    """Train LightGBM classifier on embeddings."""
    print("\nTraining downstream classifier (LightGBM)...")

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
        random_state=42,
        n_jobs=8,
        verbose=-1,
    )
    clf.fit(train_embeddings, train_targets)
    
    # Validate
    val_pred = clf.predict(val_embeddings)
    val_pred_proba = clf.predict_proba(val_embeddings)[:, 1]
    
    val_accuracy = accuracy_score(val_targets, val_pred)
    val_auc = roc_auc_score(val_targets, val_pred_proba)
    
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print(f"  Validation ROC-AUC: {val_auc:.4f}")
    
    return clf, val_accuracy, val_auc


def evaluate_on_test(clf, test_embeddings, test_targets):
    """Evaluate on test set."""
    print("\nEvaluating on test set...")
    
    test_pred = clf.predict(test_embeddings)
    test_pred_proba = clf.predict_proba(test_embeddings)[:, 1]
    
    test_accuracy = accuracy_score(test_targets, test_pred)
    test_auc = roc_auc_score(test_targets, test_pred_proba)
    
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test ROC-AUC: {test_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(test_targets, test_pred, target_names=['Non-Churn', 'Churn']))
    
    return test_accuracy, test_auc


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

    # Load data using same parquet/dataset/batch_size/max_seq_len as training
    parquet_path = Path(config['parquet_path'])
    dataset = config.get('dataset', 'churn')
    batch_size = config.get('batch_size', 64)
    max_seq_len = config.get('max_seq_len', 256)
    train_loader, val_loader, test_loader = get_dataloaders(
        parquet_path,
        dataset=dataset,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    
    # Extract embeddings
    train_emb, train_targets = extract_embeddings(model, train_loader, device)
    val_emb, val_targets = extract_embeddings(model, val_loader, device)
    test_emb, test_targets = extract_embeddings(model, test_loader, device)
    
    print("\nEmbedding shapes:")
    print(f"  Train: {train_emb.shape}")
    print(f"  Val: {val_emb.shape}")
    print(f"  Test: {test_emb.shape}")
    
    # Train downstream classifier
    clf, val_acc, val_auc = train_downstream_classifier(
        train_emb, train_targets, val_emb, val_targets
    )
    
    # Evaluate on test
    test_acc, test_auc = evaluate_on_test(clf, test_emb, test_targets)
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Validation ROC-AUC: {val_auc:.4f}")
    print(f"Test ROC-AUC:       {test_auc:.4f}")
    print(f"Test Accuracy:      {test_acc:.4f}")
    print("=" * 60)
    
    # Success criteria
    if test_auc > 0.65:
        print("✓ JEPA validation PASSED!")
        print("  (Test ROC-AUC > 0.65)")
        return True
    else:
        print("⚠ JEPA validation MARGINAL")
        print(f"  (Test ROC-AUC = {test_auc:.4f}, target > 0.65)")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)