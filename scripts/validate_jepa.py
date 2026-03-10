#!/usr/bin/env python
"""Validate JEPA by extracting embeddings and testing downstream task."""

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
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
            target = batch['target'].cpu().numpy()
            
            # Get embeddings (no masking)
            embeddings = model.get_embedding(mcc, amount)  # (batch, seq_len, emb_dim)
            
            # Global pooling (mean over sequence)
            embeddings_global = embeddings.mean(dim=1)  # (batch, emb_dim)
            
            all_embeddings.append(embeddings_global.cpu().numpy())
            all_targets.append(target)
    
    embeddings = np.concatenate(all_embeddings, axis=0)  # (total_samples, emb_dim)
    targets = np.concatenate(all_targets, axis=0)  # (total_samples,)
    
    return embeddings, targets


def train_downstream_classifier(train_embeddings, train_targets, val_embeddings, val_targets):
    """Train logistic regression classifier on embeddings."""
    print("\nTraining downstream classifier...")
    
    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
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


def main(model_path: str = None):
    """Main validation script."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # If no model path provided, find latest
    if model_path is None:
        outputs_dir = Path('outputs')
        model_dirs = sorted(outputs_dir.glob('jepa_*'))
        if not model_dirs:
            print("✗ No trained models found in outputs/")
            return
        latest_dir = model_dirs[-1]
        model_path = latest_dir / 'best_model.pt'
        print(f"Using model: {model_path}\n")
    else:
        model_path = Path(model_path)
    
    # Load model (architecture must match training: mcc_vocab_size, d_model, etc.)
    print("Loading JEPA model...")
    model = JEPA(
        mcc_vocab_size=101,
        mcc_emb_dim=24,
        d_model=1024,
        nhead=8,
        num_layers=6,
        dropout=0.1,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"  Model loaded from: {model_path}\n")
    
    # Load data (same parquet/dataset as training; align with transactions_gen_models)
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    parquet_path = data_dir / 'churn.parquet'
    dataset = 'churn'
    if not parquet_path.exists():
        parquet_path = data_dir / 'age.parquet'
        dataset = 'age'
    if not parquet_path.exists():
        parquet_path = data_dir / 'default.parquet'
        dataset = 'default'
    if not parquet_path.exists():
        parquet_path = data_dir / 'hsbc.parquet'
        dataset = 'hsbc'
    train_loader, val_loader, test_loader = get_dataloaders(
        parquet_path,
        dataset=dataset,
        batch_size=64,
        max_seq_len=256
    )
    
    # Extract embeddings
    train_emb, train_targets = extract_embeddings(model, train_loader, device)
    val_emb, val_targets = extract_embeddings(model, val_loader, device)
    test_emb, test_targets = extract_embeddings(model, test_loader, device)
    
    print(f"\nEmbedding shapes:")
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