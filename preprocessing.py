#!/usr/bin/env python
"""
Preprocessing aligned with transactions_gen_models (same datasets: churn, default, hsbc, age).

Expects flat parquet from reference repo's preprocessed/ folder:
  user_id, amount, timestamp, mcc_code, global_target
  + dataset-specific local target: churn_target (churn/hsbc), default_target (default); age has none.

Steps (mirroring config/preprocessing/*.yaml):
  1. amount -> float32
  2. Rename local target column to local_target (churn_target/default_target -> local_target)
  3. Optional: drop duplicates on (user_id, timestamp), keep first
  4. Group by user_id, sort by timestamp; output per-user records with event_time, mcc_code, amount, global_target, local_target
  5. event_time: dt_to_timestamp (unix seconds) for churn/default/hsbc, none for age
"""

from pathlib import Path
from typing import Any, Literal

import pandas as pd
import numpy as np


# Dataset configs matching transactions_gen_models config/preprocessing/*.yaml
DATASET_CONFIG = {
    "churn": {
        "local_target_col": "churn_target",
        "event_time_transform": "dt_to_timestamp",
        "drop_duplicates": False,
    },
    "churn_nodup": {
        "local_target_col": "churn_target",
        "event_time_transform": "dt_to_timestamp",
        "drop_duplicates": True,
    },
    "default": {
        "local_target_col": "default_target",
        "event_time_transform": "dt_to_timestamp",
        "drop_duplicates": False,
    },
    "default_nodup": {
        "local_target_col": "default_target",
        "event_time_transform": "dt_to_timestamp",
        "drop_duplicates": True,
    },
    "hsbc": {
        "local_target_col": "churn_target",
        "event_time_transform": "dt_to_timestamp",
        "drop_duplicates": False,
    },
    "hsbc_nodup": {
        "local_target_col": "churn_target",
        "event_time_transform": "dt_to_timestamp",
        "drop_duplicates": True,
    },
    "age": {
        "local_target_col": None,
        "event_time_transform": "none",
        "drop_duplicates": False,
    },
    "age_nodup": {
        "local_target_col": None,
        "event_time_transform": "none",
        "drop_duplicates": True,
    },
}

REQUIRED_COLUMNS = {"user_id", "amount", "timestamp", "mcc_code", "global_target"}


def _to_unix_timestamp(ser: pd.Series) -> np.ndarray:
    """Convert datetime column to Unix timestamp (seconds)."""
    return pd.to_datetime(ser).astype("datetime64[s]").astype("int64").values // 10**9


def compute_temporal_features(timestamps: pd.Series) -> tuple[list[int], list[int]]:
    """Compute day bucket and intra-day rank for a chronologically sorted sequence."""
    ts = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    calendar_day = ts.dt.normalize()
    first_day = calendar_day.iloc[0]

    time_bucket = (
        (calendar_day - first_day).dt.days.clip(lower=0, upper=1023).astype("int64").tolist()
    )
    intra_day_rank = (
        ts.groupby(calendar_day).cumcount().clip(upper=31).astype("int64").tolist()
    )
    return time_bucket, intra_day_rank


def fit_mcc_encoder(series: pd.Series, top_n: int = 100) -> dict[int, int]:
    """
    Build MCC frequency encoder: top_n most frequent codes map to 1..top_n, all others map to 0.

    Args:
        series: Raw MCC codes (e.g. df["mcc_code"]).
        top_n: Number of top codes to keep (default 100).

    Returns:
        Dict mapping raw_mcc -> encoded index (1..top_n). Codes not in the dict should be treated as 0.
    """
    counts = series.value_counts()
    top_codes = counts.head(top_n)
    return {int(mcc): idx for idx, mcc in enumerate(top_codes.index, start=1)}


def preprocess_flat_parquet(
    parquet_path: Path,
    dataset: Literal["churn", "churn_nodup", "default", "default_nodup", "hsbc", "hsbc_nodup", "age", "age_nodup"] = "churn",
) -> list[dict[str, Any]]:
    """
    Load flat parquet and convert to per-user records (same schema as transactions_gen_models after preprocess).

    Args:
        parquet_path: Path to preprocessed flat parquet (e.g. data/preprocessed/churn.parquet).
        dataset: Dataset name; determines local_target column and event_time transform.

    Returns:
        List of dicts, one per user: user_id, event_time, mcc_code, amount, global_target, local_target (optional).
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")

    cfg = DATASET_CONFIG.get(dataset)
    if cfg is None:
        cfg = {
            "local_target_col": None,
            "event_time_transform": "dt_to_timestamp",
            "drop_duplicates": False,
        }

    df = pd.read_parquet(path)

    # Required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing columns: {missing}. Expected {REQUIRED_COLUMNS}.")

    # 1. amount -> float32
    df["amount"] = df["amount"].astype("float32")

    # 2. Rename local target to local_target if present
    local_col = cfg.get("local_target_col")
    if local_col and local_col in df.columns:
        df["local_target"] = df[local_col]
    elif local_col and local_col not in df.columns:
        df["local_target"] = 0  # placeholder if column missing

    # 3. Drop duplicates (user_id, timestamp)
    if cfg.get("drop_duplicates"):
        df = df.drop_duplicates(subset=["user_id", "timestamp"], keep="first")

    # 4. event_time from timestamp (always numeric for consistency)
    if cfg.get("event_time_transform") == "dt_to_timestamp":
        df["event_time"] = _to_unix_timestamp(df["timestamp"])
    else:
        # age: no normalisation; still convert to unix seconds for consistency
        df["event_time"] = _to_unix_timestamp(df["timestamp"])

    # MCC frequency encoder: top 100 -> 1..100, rest -> 0 (before groupby, so full dataset is used)
    mcc_map = fit_mcc_encoder(df["mcc_code"], top_n=100)

    # 5. Group by user_id, sorted — matches ptls UserGroupTransformer which does sort_index()
    #    before groupby, giving records in ascending user_id order.
    df = df.sort_values(["user_id", "timestamp"])
    grouped = df.groupby("user_id", sort=False)

    records = []
    for user_id, grp in grouped:
        grp = grp.sort_values("timestamp")
        mcc_encoded = [mcc_map.get(int(m), 0) for m in grp["mcc_code"].values]
        time_bucket, intra_day_rank = compute_temporal_features(grp["timestamp"])
        rec = {
            "user_id": user_id,
            "event_time": grp["event_time"].values.tolist(),
            "mcc_code": mcc_encoded,
            "amount": grp["amount"].values.tolist(),
            "time_bucket": time_bucket,
            "intra_day_rank": intra_day_rank,
            "global_target": grp["global_target"].iloc[0],
        }
        if "local_target" in grp.columns:
            # Store as per-transaction sequence (matches ptls: local_target is NOT cols_first_item).
            # LastTokenTarget in the reference repo takes [-1] of each window slice;
            # local_target is e.g. [0,0,...,0,1,1] for a churner (last-month txns = 1).
            rec["local_target"] = grp["local_target"].astype(int).values.tolist()
        records.append(rec)

    return records


def train_val_test_split(
    records: list[dict],
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
    stratify_key: str | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split per-user records into train/val/test (same ratio as transactions_gen_models: 80/10/10).

    stratify_key: record field to stratify on. Defaults to 'local_target' so the binary
    local target is balanced across splits even for highly imbalanced datasets (e.g. churn
    ~0.6% positive rate). Set to None to disable stratification (plain random, matches the
    reference repo exactly, but can yield single-class splits on small/imbalanced data).
    Stratification is silently skipped when fewer than 2 classes are present or the key is absent.
    """
    from sklearn.model_selection import train_test_split

    def _get_strata(recs: list[dict]) -> list | None:
        if stratify_key is None:
            return None
        raw = [r.get(stratify_key) for r in recs]
        if None in raw:
            return None
        # local_target is a per-transaction list → collapse to user-level binary (any positive)
        labels = [int(any(v) if isinstance(v, list) else v) for v in raw]
        if len(set(labels)) < 2:
            return None
        return labels

    val_test_size = val_size + test_size
    train, val_test = train_test_split(
        records,
        test_size=val_test_size,
        random_state=random_state,
        stratify=_get_strata(records),
    )
    val, test = train_test_split(
        val_test,
        test_size=test_size / val_test_size,
        random_state=random_state,
        stratify=_get_strata(val_test),
    )
    return train, val, test
