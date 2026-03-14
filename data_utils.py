#!/usr/bin/env python
"""
Data loading for JEPA, aligned with transactions_gen_models preprocessing.

Supports the same datasets (churn, default, hsbc, age) and preprocessed parquet format:
  flat table: user_id, amount, timestamp, mcc_code, global_target, [churn_target|default_target for local].
We preprocess via preprocessing.preprocess_flat_parquet() and split 80/10/10 (val_size=0.1, test_size=0.1).
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Dict, List, Any, Optional, Literal

from preprocessing import (
    compute_temporal_features,
    preprocess_flat_parquet,
    train_val_test_split,
    DATASET_CONFIG,
    REQUIRED_COLUMNS,
)


# --- Dataset from preprocessed per-user records (event_time, mcc_code, amount, global_target, local_target) ---


class TransactionDataset(Dataset):
    """Dataset from per-user records (list of dicts from preprocessing)."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        max_seq_len: int = 256,
    ):
        self.records = records
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        mcc_code = rec["mcc_code"]
        amount = rec["amount"]
        time_bucket = rec.get("time_bucket")
        intra_day_rank = rec.get("intra_day_rank")
        seq_len = len(mcc_code)

        # Clamp MCC to non-negative int (padding 0)
        mccs = [max(0, int(m)) for m in mcc_code]
        amounts = [[float(a)] for a in amount]
        if time_bucket is None:
            time_buckets = list(range(seq_len))
        else:
            time_buckets = [max(0, min(1023, int(t))) for t in time_bucket]
        if intra_day_rank is None:
            intra_day_ranks = [0] * seq_len
        else:
            intra_day_ranks = [max(0, min(31, int(r))) for r in intra_day_rank]

        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            mccs = mccs + [0] * pad_len
            amounts = amounts + [[0.0]] * pad_len
            time_buckets = time_buckets + [0] * pad_len
            intra_day_ranks = intra_day_ranks + [0] * pad_len
        else:
            mccs = mccs[: self.max_seq_len]
            amounts = amounts[: self.max_seq_len]
            time_buckets = time_buckets[: self.max_seq_len]
            intra_day_ranks = intra_day_ranks[: self.max_seq_len]
            seq_len = self.max_seq_len

        out = {
            "mcc": torch.tensor(mccs, dtype=torch.long),
            "amount": torch.tensor(amounts, dtype=torch.float32),
            "time_bucket": torch.tensor(time_buckets, dtype=torch.long),
            "intra_day_rank": torch.tensor(intra_day_ranks, dtype=torch.long),
            "global_target": torch.tensor(rec["global_target"], dtype=torch.long),
            "seq_len": seq_len,
        }
        if "local_target" in rec:
            lt = rec["local_target"]
            # local_target is stored as a per-transaction list; collapse to user-level binary
            lt_scalar = int(any(lt)) if isinstance(lt, list) else int(lt)
            out["local_target"] = torch.tensor(lt_scalar, dtype=torch.long)
        else:
            out["local_target"] = torch.tensor(0, dtype=torch.long)
        return out


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    mcc_list = [b["mcc"] for b in batch]
    amount_list = [b["amount"] for b in batch]
    time_bucket_list = [b["time_bucket"] for b in batch]
    intra_day_rank_list = [b["intra_day_rank"] for b in batch]
    global_target_list = [b["global_target"] for b in batch]
    local_target_list = [b["local_target"] for b in batch]
    seq_len_list = [b["seq_len"] for b in batch]

    return {
        "mcc": torch.stack(mcc_list),
        "amount": torch.stack(amount_list),
        "time_bucket": torch.stack(time_bucket_list),
        "intra_day_rank": torch.stack(intra_day_rank_list),
        "global_target": torch.stack(global_target_list),
        "local_target": torch.stack(local_target_list),
        "seq_len": torch.tensor(seq_len_list, dtype=torch.long),
        "target": torch.stack(global_target_list),  # alias for downstream scripts
    }


# --- Backward compatibility: detect legacy parquet (one row per user with "transactions" list) ---


def _is_flat_parquet(parquet_path: Path) -> bool:
    import pandas as pd

    df_full = pd.read_parquet(parquet_path)
    return REQUIRED_COLUMNS.issubset(set(df_full.columns))


def _load_legacy_per_user_parquet(
    parquet_path: Path,
    max_seq_len: int,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load parquet that has one row per user with a 'transactions' column (list of dicts)."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    n = len(df)
    indices = np.arange(n)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    val_test_size = val_size + test_size
    train_idx = int((1 - val_test_size) * n)
    val_idx = int((1 - test_size) * n)
    train_i = indices[:train_idx]
    val_i = indices[train_idx:val_idx]
    test_i = indices[val_idx:]

    def rows_to_records(indices: np.ndarray) -> List[Dict]:
        records = []
        for i in indices:
            row = df.iloc[i]
            trans = row.get("transactions", [])
            if not trans:
                continue
            mcc_code = []
            amount = []
            timestamps = []
            for t in trans:
                if isinstance(t, dict):
                    mcc_code.append(int(t.get("mcc", t.get("mcc_code", 0))))
                    amount.append(float(t.get("amount", 0.0)))
                    timestamps.append(t.get("timestamp", t.get("event_time")))
                else:
                    mcc_code.append(0)
                    amount.append(0.0)
                    timestamps.append(None)
            if timestamps and all(ts is not None for ts in timestamps):
                time_bucket, intra_day_rank = compute_temporal_features(timestamps)
            else:
                time_bucket = list(range(len(mcc_code)))
                intra_day_rank = [0] * len(mcc_code)
            records.append({
                "user_id": row.get("user_id", str(i)),
                "mcc_code": mcc_code,
                "amount": amount,
                "time_bucket": time_bucket,
                "intra_day_rank": intra_day_rank,
                "global_target": int(row.get("global_target", row.get("target", 0))),
                "local_target": int(row.get("local_target", 0)),
            })
        return records

    train = rows_to_records(train_i)
    val = rows_to_records(val_i)
    test = rows_to_records(test_i)
    return train, val, test


# --- Local validation dataset (transactions_gen_models last-token protocol) ---

class LocalValidationDataset(Dataset):
    """Last-token validation: sequence slices with target = last value in window for target_seq_col.

    Matches config/validation/local_target.yaml and event_type.yaml + LastTokenTarget in transactions_gen_models:
    - min_len=20, random_min_seq_len=20, random_max_seq_len=40 (train)
    - window_size=32, window_step=16 (val/test, deterministic)
    - target_seq_col='local_target': target = local_target[window_end - 1] (binary label)
    - target_seq_col='mcc_code': target = mcc_code[window_end - 1] (MCC class, for event_type validation)
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        min_len: int = 20,
        random_min_seq_len: int = 20,
        random_max_seq_len: int = 40,
        window_size: int = 32,
        window_step: int = 16,
        deterministic: bool = False,
        max_seq_len: int = 40,
        random_state: Optional[int] = None,
        target_seq_col: Literal["local_target", "mcc_code"] = "local_target",
    ):
        self.target_seq_col = target_seq_col
        if target_seq_col == "local_target":
            self.records = [r for r in records if len(r["mcc_code"]) >= min_len and "local_target" in r]
        else:
            self.records = [r for r in records if len(r["mcc_code"]) >= min_len]
        self.min_len = min_len
        self.random_min = random_min_seq_len
        self.random_max = random_max_seq_len
        self.window_size = window_size
        self.window_step = window_step
        self.deterministic = deterministic
        self.max_seq_len = max_seq_len
        self.rng = np.random.default_rng(random_state)
        # Precompute sliding-window indices for deterministic (val/test)
        self._windows: List[Tuple[int, int, int]] = []  # (record_idx, start, end)
        if deterministic:
            for i, rec in enumerate(self.records):
                L = len(rec["mcc_code"])
                for start in range(0, max(0, L - window_size), window_step):
                    end = start + window_size
                    if end <= L and end - start >= min_len:
                        self._windows.append((i, start, end))

    def __len__(self) -> int:
        if self.deterministic:
            return len(self._windows)
        return len(self.records)

    def _get_slice(self, rec: Dict[str, Any], start: int, end: int) -> Dict[str, torch.Tensor]:
        seq_len = end - start
        mccs = [max(0, int(m)) for m in rec["mcc_code"][start:end]]
        amounts = [[float(a)] for a in rec["amount"][start:end]]
        time_bucket = rec.get("time_bucket")
        intra_day_rank = rec.get("intra_day_rank")
        if time_bucket is not None:
            time_buckets = [max(0, min(1023, int(t))) for t in time_bucket[start:end]]
        else:
            time_buckets = list(range(seq_len))
        if intra_day_rank is not None:
            intra_day_ranks = [max(0, min(31, int(r))) for r in intra_day_rank[start:end]]
        else:
            intra_day_ranks = [0] * seq_len
        pad_len = self.max_seq_len - seq_len
        if pad_len > 0:
            mccs = mccs + [0] * pad_len
            amounts = amounts + [[0.0]] * pad_len
            time_buckets = time_buckets + [0] * pad_len
            intra_day_ranks = intra_day_ranks + [0] * pad_len
        else:
            mccs = mccs[: self.max_seq_len]
            amounts = amounts[: self.max_seq_len]
            time_buckets = time_buckets[: self.max_seq_len]
            intra_day_ranks = intra_day_ranks[: self.max_seq_len]
        out = {
            "mcc": torch.tensor(mccs, dtype=torch.long),
            "amount": torch.tensor(amounts, dtype=torch.float32),
            "time_bucket": torch.tensor(time_buckets, dtype=torch.long),
            "intra_day_rank": torch.tensor(intra_day_ranks, dtype=torch.long),
        }
        if self.target_seq_col == "local_target":
            # (matches LastTokenTarget: target = x["local_target"][-1])
            local_tgt_seq = rec["local_target"]
            local_tgt = float(local_tgt_seq[end - 1]) if isinstance(local_tgt_seq, list) else float(local_tgt_seq)
            out["local_target"] = torch.tensor(local_tgt, dtype=torch.float32)
        else:
            # target_seq_col == "mcc_code": last MCC in window (event_type validation)
            last_mcc = int(rec["mcc_code"][end - 1])
            out["mcc_target"] = torch.tensor(last_mcc, dtype=torch.long)
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.deterministic:
            i, start, end = self._windows[idx]
            return self._get_slice(self.records[i], start, end)
        rec = self.records[idx]
        L = len(rec["mcc_code"])
        crop_len = int(self.rng.integers(self.random_min, self.random_max + 1))
        crop_len = min(crop_len, L)
        start = int(self.rng.integers(0, max(1, L - crop_len + 1)))
        end = start + crop_len
        return self._get_slice(rec, start, end)


def collate_local_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "mcc": torch.stack([b["mcc"] for b in batch]),
        "amount": torch.stack([b["amount"] for b in batch]),
        "time_bucket": torch.stack([b["time_bucket"] for b in batch]),
        "intra_day_rank": torch.stack([b["intra_day_rank"] for b in batch]),
        "local_target": torch.stack([b["local_target"] for b in batch]),
    }


def collate_mcc_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate for last-token MCC (event_type) validation."""
    return {
        "mcc": torch.stack([b["mcc"] for b in batch]),
        "amount": torch.stack([b["amount"] for b in batch]),
        "time_bucket": torch.stack([b["time_bucket"] for b in batch]),
        "intra_day_rank": torch.stack([b["intra_day_rank"] for b in batch]),
        "mcc_target": torch.stack([b["mcc_target"] for b in batch]),
    }


# --- Public API ---

DatasetName = Literal[
    "churn", "churn_nodup", "default", "default_nodup", "hsbc", "hsbc_nodup", "age", "age_nodup"
]


def get_preprocessed_splits(
    parquet_path: Path,
    dataset: Optional[DatasetName] = "churn",
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (train_rec, val_rec, test_rec) from parquet using reference preprocessing/split."""
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    if dataset and dataset in DATASET_CONFIG:
        records = preprocess_flat_parquet(parquet_path, dataset=dataset)
        return train_val_test_split(
            records, val_size=val_size, test_size=test_size, random_state=random_state
        )
    # Legacy: per-user parquet
    train_rec, val_rec, test_rec = _load_legacy_per_user_parquet(
        parquet_path, max_seq_len=256, val_size=val_size, test_size=test_size, random_state=random_state
    )
    return train_rec, val_rec, test_rec


def get_dataloaders(
    parquet_path: Path,
    dataset: Optional[DatasetName] = "churn",
    batch_size: int = 32,
    max_seq_len: int = 256,
    num_workers: int = 0,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train/val/test dataloaders from parquet, using same preprocessing as transactions_gen_models.

    Args:
        parquet_path: Path to parquet (flat preprocessed: user_id, amount, timestamp, mcc_code, global_target, ...).
        dataset: One of churn, churn_nodup, default, default_nodup, hsbc, hsbc_nodup, age, age_nodup.
                 Determines local_target column and optional dedup. If None, tries legacy per-user parquet.
        batch_size: Batch size.
        max_seq_len: Max sequence length (pad/truncate).
        num_workers: DataLoader workers.
        val_size: Validation fraction (default 0.1).
        test_size: Test fraction (default 0.1).
        random_state: Split seed (default 42).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    parquet_path = Path(parquet_path)
    train_rec, val_rec, test_rec = get_preprocessed_splits(
        parquet_path, dataset=dataset, val_size=val_size, test_size=test_size, random_state=random_state
    )
    print(f"\nLoaded {parquet_path} (dataset={dataset or 'legacy'})")
    print(f"  Users: train={len(train_rec)}, val={len(val_rec)}, test={len(test_rec)}")

    train_ds = TransactionDataset(train_rec, max_seq_len=max_seq_len)
    val_ds = TransactionDataset(val_rec, max_seq_len=max_seq_len)
    test_ds = TransactionDataset(test_rec, max_seq_len=max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    project_root = Path(__file__).parent
    parquet_path = project_root / "data" / "churn.parquet"
    if not parquet_path.exists():
        parquet_path = project_root / "data" / "age.parquet"

    print("Testing data loading (reference preprocessing)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        parquet_path,
        dataset="churn" if "churn" in str(parquet_path) else "age",
        batch_size=4,
    )
    for batch in train_loader:
        print("\nBatch shapes:")
        print("  mcc:", batch["mcc"].shape)
        print("  amount:", batch["amount"].shape)
        print("  global_target:", batch["global_target"].shape)
        print("  seq_len:", batch["seq_len"].shape)
        break
    print(f"\nTrain batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
