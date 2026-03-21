# LatentLedger

JEPA training on transaction sequences with [Hydra](https://hydra.cc/) configs and [Weights & Biases](https://wandb.ai/) logging.

## Setup

From the repository root:

```bash
uv sync
```

Place dataset parquets under `data/` (e.g. `data/churn.parquet`). Paths are configured in `configs/dataset/*.yaml`.

Optional: verify files exist:

```bash
uv run python scripts/download_data.py
```

Log in to W&B if you use it:

```bash
wandb login
```

Run all commands below from the **repository root** so paths like `data/churn.parquet` resolve correctly.

---

## Smoke test (single run)

Quick check on one dataset and a couple of epochs:

```bash
uv run python scripts/train_jepa_hydra.py \
  dataset=churn \
  model=small \
  training.epochs=2 \
  training.batch_size=32
```

Validate the **latest** checkpoint under `outputs/` (most recently written `best_model.pt`):

```bash
uv run python scripts/validate_jepa_hydra.py
```

Validate a **specific** training run directory (contains `config.json` and `best_model.pt`):

```bash
uv run python scripts/validate_jepa_hydra.py experiment_dir=outputs/YYYY-MM-DD/HH-MM-SS
```

Hydra prints the run directory when training starts; use that path for `experiment_dir`.

---

## Full benchmark (main grid)

This matches the experiment matrix in `EXPERIMENT_PLAN.md`: four datasets × two model sizes (small, base) × temporal vs ordinal × three seeds.

**Training** (48 Hydra jobs; Age with `use_temporal_encoding=true` falls back to ordinal and logs a warning, so **42** meaningfully distinct training configs):

```bash
uv run python scripts/train_jepa_hydra.py --multirun \
  dataset=churn,default,hsbc,age \
  model=small,base \
  use_temporal_encoding=true,false \
  seed=42,123,456
```

**Validation** is not keyed to a specific checkpoint by dataset/seed. The default is to load the **single latest** `best_model.pt` under `outputs/`. To evaluate **every** training output after a sweep, run validation once per run directory, for example:

```bash
find outputs -name best_model.pt | while read f; do
  d="$(dirname "$f")"
  echo "=== $d ==="
  uv run python scripts/validate_jepa_hydra.py "experiment_dir=$d"
done
```

Or one run at a time:

```bash
uv run python scripts/validate_jepa_hydra.py experiment_dir=outputs/<your-run-folder>
```

---

## Age-only: large model (1024-d)

The `large` config is intended only for the Age dataset (enough volume to avoid collapse). Example:

```bash
uv run python scripts/train_jepa_hydra.py \
  dataset=age \
  model=large \
  use_temporal_encoding=false \
  seed=42
```

You can sweep seeds the same way as the main grid:

```bash
uv run python scripts/train_jepa_hydra.py --multirun \
  dataset=age \
  model=large \
  use_temporal_encoding=false \
  seed=42,123,456
```

---

## Useful overrides

| Goal | Example |
|------|--------|
| Change epochs | `training.epochs=60` |
| Batch size | `training.batch_size=128` |
| DataLoader workers | `training.num_workers=4` |
| Disable cosine LR | `training.use_cosine_annealing=false` |
| CPU only | `device=cpu` |

Legacy argparse entrypoint (composes the same YAML without Hydra):

```bash
uv run python scripts/train_jepa.py --dataset churn --model-size small --epochs 2
```

---

## Experiment reference

Full grid, run naming pattern, and W&B group `benchmark_v1` are documented in [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md).
