#!/usr/bin/env python
"""Train JEPA on Churn dataset."""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.jepa import JEPA
from data_utils import get_dataloaders


def train_epoch(model, train_loader, optimizer, device, epoch, total_steps, steps_completed):
    """Train for one epoch. EMA decay uses global step (Appendix A.1)."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")

    for batch_idx, batch in enumerate(pbar):
        mcc = batch['mcc'].to(device)
        amount = batch['amount'].to(device)

        loss, _ = model(mcc, amount)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        step = steps_completed + batch_idx
        tau = model.get_ema_decay(step, total_steps)
        model._update_target_encoder(tau=tau)

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / num_batches


def validate(model, val_loader, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            mcc = batch['mcc'].to(device)
            amount = batch['amount'].to(device)
            
            loss, _ = model(mcc, amount)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """Main training script."""
    
    # Config (datasets match transactions_gen_models: churn, default, hsbc, age)
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    default_parquet = data_dir / 'churn.parquet'
    default_dataset = 'churn'
    if not default_parquet.exists():
        default_parquet = data_dir / 'age.parquet'
        default_dataset = 'age'
    if not default_parquet.exists():
        default_parquet = data_dir / 'default.parquet'
        default_dataset = 'default'
    if not default_parquet.exists():
        default_parquet = data_dir / 'hsbc.parquet'
        default_dataset = 'hsbc'

    config = {
        'batch_size': 32,
        'max_seq_len': 256,
        'mcc_vocab_size': 101,  # Paper: top-100 MCCs + 1 mask token (0). Use 101 for Churn/HSBC/Default/Age.
        'mcc_emb_dim': 24,  # 24 for Churn/HSBC, 16 for Age/Default (Appendix A)
        'd_model': 1024,
        'nhead': 8,
        'num_layers': 6,
        'predictor_d_model': 384,  # narrow predictor bottleneck (I-JEPA; must be divisible by nhead if predictor nhead used)
        'dropout': 0.1,
        'learning_rate': 1e-3,
        'epochs': 5,  # Start small for validation
        'parquet_path': str(default_parquet),
        'dataset': default_dataset,  # churn | default | hsbc | age (same as reference repo)
        'device': (
            'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available()
            else 'cpu'
        ),
    }
    
    print("=" * 60)
    print("JEPA Training Configuration")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = output_dir / f'jepa_{timestamp}'
    exp_dir.mkdir(exist_ok=True)
    
    # Save config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Setup tensorboard
    writer = SummaryWriter(str(exp_dir / 'logs'))
    
    # Create model
    print("\nInitializing JEPA model...")
    model = JEPA(
        mcc_vocab_size=config['mcc_vocab_size'],
        mcc_emb_dim=config['mcc_emb_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        predictor_d_model=config['predictor_d_model'],
        dropout=config['dropout'],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate']
    )
    
    # Data loaders (preprocessing aligned with transactions_gen_models)
    print("\nLoading data...")
    parquet_path = Path(config['parquet_path'])
    train_loader, val_loader, test_loader = get_dataloaders(
        parquet_path,
        dataset=config.get('dataset', 'churn'),
        batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len']
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    total_steps = len(train_loader) * config['epochs']
    print("\nStarting training...\n")

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    steps_completed = 0

    for epoch in range(1, config['epochs'] + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, total_steps, steps_completed
        )
        steps_completed += len(train_loader)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Log
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f"Epoch {epoch}/{config['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), exp_dir / 'best_model.pt')
            print(f"  → Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch} epochs")
                break
    
    writer.close()
    
    # Test
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(exp_dir / 'best_model.pt'))
    test_loss = validate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save results
    results = {
        'config': config,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'epochs_trained': epoch,
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"  Outputs saved to: {exp_dir}")
    print(f"  Best model: {exp_dir / 'best_model.pt'}")
    
    return exp_dir


if __name__ == "__main__":
    exp_dir = main()