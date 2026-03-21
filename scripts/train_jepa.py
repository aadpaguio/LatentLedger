#!/usr/bin/env python
"""Train JEPA (legacy argparse entrypoint; composes the same YAML as Hydra)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from jepa_training import load_composed_config, run_training


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train JEPA. Options override composed defaults from configs/.",
    )
    parser.add_argument("--parquet-path", type=str, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["churn", "default", "hsbc", "age", "churn_nodup", "default_nodup", "hsbc_nodup", "age_nodup"],
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--mcc-vocab-size", type=int, default=None)
    parser.add_argument("--mcc-emb-dim", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--predictor-d-model", type=int, default=None)
    parser.add_argument("--predictor-num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--use-temporal-encoding",
        action="store_true",
        help="Use time-aware positional embeddings",
    )
    parser.add_argument("--no-temporal-encoding", action="store_true", help="Force ordinal positions")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--no-cosine-annealing", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "base", "large"],
        help="Which configs/model/*.yaml to load (default: small).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset or "churn"
    cfg = load_composed_config(dataset=dataset_name, model=args.model_size, training_name="default")

    if args.seed is not None:
        cfg.seed = args.seed
    if args.use_temporal_encoding:
        cfg.use_temporal_encoding = True
    if args.no_temporal_encoding:
        cfg.use_temporal_encoding = False
    if args.device is not None:
        cfg.device = args.device
    if args.parquet_path is not None:
        cfg.dataset.parquet_path = args.parquet_path
    elif args.dataset is not None:
        root = Path(__file__).resolve().parent.parent
        stem = args.dataset.replace("_nodup", "")
        cfg.dataset.parquet_path = str(root / "data" / f"{stem}.parquet")
    if args.max_seq_len is not None:
        cfg.dataset.max_seq_len = args.max_seq_len
    if args.mcc_vocab_size is not None:
        cfg.dataset.mcc_vocab_size = args.mcc_vocab_size
    if args.mcc_emb_dim is not None:
        cfg.dataset.mcc_emb_dim = args.mcc_emb_dim
    if args.d_model is not None:
        cfg.model.d_model = args.d_model
    if args.nhead is not None:
        cfg.model.nhead = args.nhead
    if args.num_layers is not None:
        cfg.model.num_layers = args.num_layers
    if args.predictor_d_model is not None:
        cfg.model.predictor_d_model = args.predictor_d_model
    if args.predictor_num_layers is not None:
        cfg.model.predictor_num_layers = args.predictor_num_layers
    if args.dropout is not None:
        cfg.model.dropout = args.dropout
    if args.learning_rate is not None:
        cfg.training.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        cfg.training.weight_decay = args.weight_decay
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.no_cosine_annealing:
        cfg.training.use_cosine_annealing = False

    OmegaConf.resolve(cfg)
    print("=" * 60)
    print("JEPA Training Configuration (composed YAML)")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    return run_training(cfg, hydra_output_dir=None)


if __name__ == "__main__":
    main()
