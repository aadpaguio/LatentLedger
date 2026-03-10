# Data (aligned with transactions_gen_models)

This repo uses the **same datasets and preprocessing** as [transactions_gen_models](https://github.com/romanenkova95/transactions_gen_models).

## Where to get the data

- **Source**: [Yandex Disk](https://disk.yandex.ru/d/--KIPMEJ-cB4MA) — download the `preprocessed/` folder.
- Place the parquet files in `data/`:
  - `data/churn.parquet`
  - `data/default.parquet`
  - `data/hsbc.parquet`
  - `data/age.parquet`

## Expected parquet format (flat, preprocessed)

Each file is a **flat table** (one row per transaction) with columns:

| Column         | Description                          |
|----------------|--------------------------------------|
| `user_id`      | User identifier                      |
| `amount`       | Transaction amount (numeric)        |
| `timestamp`    | Transaction time (datetime or unix) |
| `mcc_code`     | MCC category code                   |
| `global_target`| User-level target (e.g. churn 0/1)   |

**Dataset-specific local target** (optional, for downstream tasks):

- **churn**, **hsbc**: column `churn_target` → used as `local_target`
- **default**: column `default_target` → used as `local_target`
- **age**: no local target column

## Preprocessing (what we do)

1. Cast `amount` to float32.
2. Rename dataset-specific target to `local_target` (churn_target / default_target).
3. Optional: drop duplicates on `(user_id, timestamp)` for `*_nodup` configs.
4. Group by `user_id`, sort by `timestamp`; produce per-user sequences.
5. Train/val/test split: **80% / 10% / 10%**, `random_state=42`.

Dataset names: `churn`, `churn_nodup`, `default`, `default_nodup`, `hsbc`, `hsbc_nodup`, `age`, `age_nodup`.

## Config reference

| Dataset  | Local target column | Dedup (user_id, timestamp) |
|----------|---------------------|----------------------------|
| churn    | churn_target        | no                         |
| churn_nodup | churn_target     | yes                        |
| default  | default_target      | no                         |
| default_nodup | default_target   | yes                        |
| hsbc     | churn_target        | no                         |
| hsbc_nodup | churn_target      | yes                        |
| age      | —                   | no                         |
| age_nodup | —                  | yes                        |
