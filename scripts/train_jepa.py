#!/usr/bin/env python
"""Train JEPA on Churn dataset."""

import argparse
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.jepa import JEPA
from data_utils import get_dataloaders
from diagnostics import compute_collapse_diagnostics


def parse_args():
    """Parse CLI args; only overrides config for explicitly provided options."""
    parser = argparse.ArgumentParser(
        description="Train JEPA. All options override defaults from config."
    )
    # Data
    parser.add_argument("--parquet-path", type=str, default=None, help="Path to parquet file")
    parser.add_argument("--dataset", type=str, default=None, choices=["churn", "default", "hsbc", "age"], help="Dataset name")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Max sequence length")
    # Model
    parser.add_argument("--mcc-vocab-size", type=int, default=None, help="MCC vocab size (e.g. 101)")
    parser.add_argument("--mcc-emb-dim", type=int, default=None, help="MCC embedding dim")
    parser.add_argument("--d-model", type=int, default=None, help="Transformer d_model")
    parser.add_argument("--nhead", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=None, help="Encoder num layers")
    parser.add_argument("--predictor-d-model", type=int, default=None, help="Predictor bottleneck dim")
    parser.add_argument("--predictor-num-layers", type=int, default=None, help="Number of predictor transformer layers")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
    parser.add_argument(
        "--use-temporal-encoding",
        action="store_true",
        help="Use time-aware positional embeddings instead of ordinal position embeddings",
    )
    # Training
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Adam weight decay (default: 0.4)")
    parser.add_argument("--no-cosine-annealing", action="store_true", help="Use constant LR instead of cosine annealing")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"], help="Device to train on")
    return parser.parse_args()


def train_epoch(model, train_loader, optimizer, device, epoch, total_steps, steps_completed):
    """Train for one epoch. EMA decay uses global step (Appendix A.1)."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")

    for batch_idx, batch in enumerate(pbar):
        mcc = batch['mcc'].to(device)
        amount = batch['amount'].to(device)
        time_bucket = batch['time_bucket'].to(device)
        intra_day_rank = batch['intra_day_rank'].to(device)

        loss, _ = model(
            mcc,
            amount,
            time_bucket=time_bucket,
            intra_day_rank=intra_day_rank,
        )

        optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        step = steps_completed + batch_idx
        tau = model.get_ema_decay(step, total_steps)
        model._update_target_encoder(tau=tau)

        wandb.log({"train/step_loss": loss.item(), "train/grad_norm": total_norm}, step=step)
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / num_batches


def validate(model, val_loader, device, debug=True):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            mcc = batch['mcc'].to(device)
            amount = batch['amount'].to(device)
            time_bucket = batch['time_bucket'].to(device)
            intra_day_rank = batch['intra_day_rank'].to(device)
            
            loss, sx = model(
                mcc,
                amount,
                time_bucket=time_bucket,
                intra_day_rank=intra_day_rank,
            )
            
            total_loss += loss.item()
            num_batches += 1

            if debug and batch_idx == 0:
                # Probe target encoder directly
                sy = model.target_encoder(
                    mcc,
                    amount,
                    time_bucket=time_bucket,
                    intra_day_rank=intra_day_rank,
                )  # (batch, seq_len, d_model)
                
                print("\n  [Debug - first val batch]")
                print(f"  Target encoder output norm:   {sy.norm(dim=-1).mean():.4f}")
                print(f"  Context encoder output norm:  {sx.norm(dim=-1).mean():.4f}")
                print(f"  Target std across batch:      {sy.std(dim=0).mean():.6f}")
                print(f"  Target mean across batch:     {sy.mean():.6f}")
                print(f"  Loss this batch:              {loss.item():.6f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """Main training script."""
    args = parse_args()

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

    default_device = (
        'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )

    config = {
        'batch_size': 32,
        'max_seq_len': 256,
        'mcc_vocab_size': 101,  # Paper: top-100 MCCs + 1 mask token (0). Use 101 for Churn/HSBC/Default/Age.
        'mcc_emb_dim': 24,  # 24 for Churn/HSBC, 16 for Age/Default (Appendix A)
        'd_model': 256,
        'nhead': 4,
        'num_layers': 4,
        'predictor_d_model': 96,  # narrow predictor bottleneck (I-JEPA; must be divisible by nhead if predictor nhead used)
        'predictor_num_layers': 12,  # number of transformer layers in the predictor
        'dropout': 0.2,
        'learning_rate': 3e-4,
        'weight_decay': 0.4,
        'use_cosine_annealing': True,
        'epochs': 5,  # Start small for validation
        'parquet_path': str(default_parquet),
        'dataset': default_dataset,  # churn | default | hsbc | age (same as reference repo)
        'device': default_device,
        'use_temporal_encoding': False,
        # Data split (match transactions_gen_models config/preprocessing/*.yaml)
        'val_size': 0.1,
        'test_size': 0.1,
        'random_state': 42,
    }

    # Override config from CLI (only keys that were explicitly passed)
    cli_map = {
        'parquet_path': args.parquet_path,
        'dataset': args.dataset,
        'batch_size': args.batch_size,
        'max_seq_len': args.max_seq_len,
        'mcc_vocab_size': args.mcc_vocab_size,
        'mcc_emb_dim': args.mcc_emb_dim,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'predictor_d_model': args.predictor_d_model,
        'predictor_num_layers': args.predictor_num_layers,
        'dropout': args.dropout,
        'use_temporal_encoding': True if args.use_temporal_encoding else None,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'device': args.device,
    }
    for key, value in cli_map.items():
        if value is not None:
            config[key] = value
    if args.no_cosine_annealing:
        config['use_cosine_annealing'] = False

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
    
    wandb.init(project="latentledger", job_type="training", config=config, name=exp_dir.name)
    
    # Create model
    print("\nInitializing JEPA model...")
    model = JEPA(
        mcc_vocab_size=config['mcc_vocab_size'],
        mcc_emb_dim=config['mcc_emb_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        predictor_d_model=config['predictor_d_model'],
        predictor_num_layers=config.get('predictor_num_layers', 12),
        dropout=config['dropout'],
        use_temporal_encoding=config.get('use_temporal_encoding', False),
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    # LR schedule: cosine annealing (optional)
    scheduler = None
    if config['use_cosine_annealing']:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=1e-6
        )

    # Data loaders (preprocessing aligned with transactions_gen_models)
    print("\nLoading data...")
    parquet_path = Path(config['parquet_path'])
    train_loader, val_loader, test_loader = get_dataloaders(
        parquet_path,
        dataset=config.get('dataset', 'churn'),
        batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len'],
        val_size=config.get('val_size', 0.1),
        test_size=config.get('test_size', 0.1),
        random_state=config.get('random_state', 42),
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
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else config['learning_rate']
        wandb.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "lr": current_lr,
        }, step=steps_completed)
        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch}/{config['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), exp_dir / 'best_model.pt')
            print("  → Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch} epochs")
                break
        
        diag = compute_collapse_diagnostics(model, val_loader, device, n_batches=3)
        wandb.log(diag, step=steps_completed)
        print(f"  eff_rank={diag['diagnostics/effective_rank']:.1f}  "
              f"cos_sim={diag['diagnostics/encoder_cosine_sim']:.4f}  "
              f"probe_auc={diag['diagnostics/probe_roc_auc']:.3f}  "
              f"param_l2={diag['diagnostics/param_l2_distance']:.2f}")

    wandb.finish()
    
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
    
    print("\n✓ Training complete!")
    print(f"  Outputs saved to: {exp_dir}")
    print(f"  Best model: {exp_dir / 'best_model.pt'}")
    
    return exp_dir


if __name__ == "__main__":
    exp_dir = main()