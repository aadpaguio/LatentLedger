# LatentLedger: Experiment Formalization Plan

---

## 1. Experiment Infrastructure: Hydra + W&B

### Why Hydra

Your current setup uses a flat `config` dict in `train_jepa.py` with CLI overrides via argparse. This works for one-off runs but makes it painful to:
- Track which hyperparameters produced which results
- Launch sweeps across datasets × temporal encoding × model sizes
- Reproduce a run from its config alone

Hydra gives you structured YAML configs, overrides from the command line, automatic output directory management, and a `--multirun` flag that lets you launch your full experiment matrix in one command. Combined with W&B, every run gets a unique, descriptive name and all configs are logged automatically.

### Proposed Config Structure

```
configs/
├── config.yaml              # defaults list, pulls in dataset + model + training
├── dataset/
│   ├── churn.yaml
│   ├── default.yaml
│   ├── hsbc.yaml
│   └── age.yaml
├── model/
│   ├── small.yaml           # d_model=256 (current working config)
│   ├── base.yaml            # d_model=512
│   └── large.yaml           # d_model=1024 (paper-matched MLM)
├── training/
│   └── default.yaml         # lr, weight_decay, epochs, scheduler, EMA
└── experiment/
    ├── full_benchmark.yaml   # multirun sweep definition
    └── ablation_temporal.yaml
```

### `configs/config.yaml`
```yaml
defaults:
  - dataset: churn
  - model: base
  - training: default
  - _self_

seed: 42
use_temporal_encoding: true

wandb:
  project: latentledger
  tags: ["jepa", "${dataset.name}"]
```

### `configs/dataset/churn.yaml`
```yaml
name: churn
parquet_path: data/churn.parquet
mcc_emb_dim: 24          # Paper: 24 for Churn/HSBC
mcc_vocab_size: 101       # top-100 + mask
has_local_target: true
has_timestamps: true      # can use temporal encoding
max_seq_len: 256
```

### `configs/dataset/age.yaml`
```yaml
name: age
parquet_path: data/age.parquet
mcc_emb_dim: 16          # Paper: 16 for Age/Default
mcc_vocab_size: 101
has_local_target: false
has_timestamps: false     # serial day numbers only, no real timestamps
max_seq_len: 256
```

### `configs/dataset/default.yaml`
```yaml
name: default
parquet_path: data/default.parquet
mcc_emb_dim: 16          # Paper: 16 for Age/Default
mcc_vocab_size: 101
has_local_target: true
has_timestamps: true
max_seq_len: 256
```

### `configs/dataset/hsbc.yaml`
```yaml
name: hsbc
parquet_path: data/hsbc.parquet
mcc_emb_dim: 24          # Paper: 24 for Churn/HSBC
mcc_vocab_size: 101
has_local_target: true
has_timestamps: true
max_seq_len: 256
```

### `configs/model/base.yaml` (target: match MLM baseline scale)
```yaml
# Matches paper MLM: Transformer, 6 layers, 8 heads, d_model=1024
# But we found d_model=1024 collapses on small datasets
# base = 512, the sweet spot between expressiveness and collapse risk
size_name: base
d_model: 512
nhead: 8
num_layers: 6
predictor_d_model: 192   # narrow bottleneck, ~d_model/2.67
predictor_num_layers: 6   # smaller encoder → shallower predictor (I-JEPA Appendix A.1)
dropout: 0.1
```

### `configs/model/small.yaml` (your proven working config)
```yaml
size_name: small
d_model: 256
nhead: 4
num_layers: 4
predictor_d_model: 96
predictor_num_layers: 6
dropout: 0.2
```

### `configs/model/large.yaml` (paper-matched MLM, collapse-prone)
```yaml
# Direct match to paper MLM: 6 layers, 8 heads, d=1024
# Only viable on Age (30K users, 26M transactions)
size_name: large
d_model: 1024
nhead: 8
num_layers: 6
predictor_d_model: 384    # I-JEPA standard predictor dim
predictor_num_layers: 12
dropout: 0.1
```

### `configs/training/default.yaml`
```yaml
learning_rate: 3e-4
weight_decay: 0.4         # I-JEPA uses 0.04→0.4 annealing; we use fixed 0.4
epochs: 60                # match CoLES/AR training budget
batch_size: 128           # match paper baselines (CoLES, AR, NHP)
use_cosine_annealing: true
ema_tau_start: 0.90       # critical: must start low
ema_tau_end: 0.999
grad_clip_max_norm: 1.0
val_size: 0.1
test_size: 0.1
```

### Naming Convention

**W&B run name**: `{dataset}_{model_size}_{temporal|ordinal}_{seed}`

Examples:
- `churn_base_temporal_42`
- `age_small_ordinal_42`
- `default_base_temporal_123`

**W&B tags**: `["jepa", "{dataset}", "{model_size}", "{temporal|ordinal}"]`

**W&B group**: `benchmark_v1` (all runs in this campaign share a group for easy comparison)

### Multirun Command (all experiments in one go)

```bash
python scripts/train_jepa_hydra.py --multirun \
  dataset=churn,default,hsbc,age \
  model=small,base \
  use_temporal_encoding=true,false \
  seed=42,123,456
```

This launches **4 datasets × 2 model sizes × 2 encoding modes × 3 seeds = 48 runs**.

For Age with `use_temporal_encoding=true`, the script should auto-fallback to `false` and log a warning (since Age lacks real timestamps). This brings effective runs to **42 unique configs**.

After training, validation runs as a separate sweep:

```bash
python scripts/validate_jepa_hydra.py --multirun \
  dataset=churn,default,hsbc,age \
  model=small,base \
  use_temporal_encoding=true,false \
  seed=42,123,456
```

### Key Changes to `train_jepa.py`

1. **Replace argparse with `@hydra.main`**
2. **Replace flat config dict with OmegaConf DictConfig**
3. **Auto-resolve temporal encoding**: `if cfg.use_temporal_encoding and not cfg.dataset.has_timestamps: cfg.use_temporal_encoding = False`
4. **Structured W&B init**:
   ```python
   wandb.init(
       project=cfg.wandb.project,
       group="benchmark_v1",
       name=f"{cfg.dataset.name}_{cfg.model.size_name}_{'temporal' if cfg.use_temporal_encoding else 'ordinal'}_{cfg.seed}",
       tags=cfg.wandb.tags,
       config=OmegaConf.to_container(cfg, resolve=True),
   )
   ```
5. **Seed everything**: `torch.manual_seed(cfg.seed)`, `np.random.seed(cfg.seed)`, etc.

---

## 2. Experiment Matrix & Hyperparameters

### Model Size Rationale

The reference paper baselines use these embedding dimensions on public datasets:

| Method | Type          | Embedding dim | Layers | Notes                     |
|--------|---------------|---------------|--------|---------------------------|
| MLM    | Transformer   | 1024          | 6      | 8 heads, ffn=1024         |
| AE/AR  | LSTM          | 1024          | 1      | Hidden dim                |
| CoLES  | LSTM          | 1024          | 1      | Churn/Age/HSBC; 800 for Default |
| TS2Vec | 1D Conv       | 1024          | 10     | Output channels           |

Your JEPA at `d_model=256` already beat baselines on Churn. To make a fair comparison, you need to also test at scale closer to the baselines (512 and potentially 1024), while being aware that larger models need more data to avoid collapse.

### Full Experiment Grid

| Dataset | Model    | d_model | Temporal | Seeds     | Epochs | Batch |
|---------|----------|---------|----------|-----------|--------|-------|
| Churn   | small    | 256     | yes/no   | 42,123,456| 60     | 128   |
| Churn   | base     | 512     | yes/no   | 42,123,456| 60     | 128   |
| Default | small    | 256     | yes/no   | 42,123,456| 60     | 128   |
| Default | base     | 512     | yes/no   | 42,123,456| 60     | 128   |
| HSBC    | small    | 256     | yes/no   | 42,123,456| 60     | 128   |
| HSBC    | base     | 512     | yes/no   | 42,123,456| 60     | 128   |
| Age     | small    | 256     | no       | 42,123,456| 60     | 128   |
| Age     | base     | 512     | no       | 42,123,456| 60     | 128   |
| Age     | large    | 1024    | no       | 42,123,456| 60     | 128   |

**Why `large` only on Age**: Age has 30K users / 26M transactions — enough data to possibly sustain d_model=1024 without collapse. The other datasets (4-7K users) showed collapse at 1024 in your experiments. If `large` on Age works well, that's interesting data for the blog post about capacity-data scaling.

**Why VICReg is removed**: You explored it and deferred it. The current JEPA loss + EMA schedule is stable. Adding VICReg would mean 2 extra hyperparameters (cov_weight, var_weight) that multiply the sweep. Remove it from the model and config to keep the experiment clean. You can always add it as a follow-up ablation.

### Hyperparameters to Sweep (per model size)

**Fixed across all runs (no sweeping)**:
- `mcc_vocab_size`: 101 (all datasets)
- `mcc_emb_dim`: 24 (Churn/HSBC), 16 (Default/Age) — **set per-dataset, not swept**
- `max_seq_len`: 256
- `ema_tau_start`: 0.90, `ema_tau_end`: 0.999
- `weight_decay`: 0.4
- `grad_clip`: 1.0
- `val_size / test_size`: 0.1 / 0.1
- `random_state` for data split: 42 (always, for reproducibility across seeds)
- `use_cosine_annealing`: true
- `batch_size`: 128

**Swept**:
- `model`: small, base (+ large for Age)
- `use_temporal_encoding`: true, false (false only for Age)
- `seed`: 42, 123, 456

### Not Swept (held fixed, justified)

- **learning_rate=3e-4**: This worked well across your experiments. The I-JEPA paper uses lr warmup + decay; your cosine annealing approximates this. Not worth sweeping given compute budget.
- **epochs=60**: Matches CoLES/AR training budget. AE/MLM use 2000 steps at batch=1024 which is roughly similar. If Age takes too long, you can monitor val loss curves and early-stop.
- **predictor_num_layers**: Scales with model size (6 for small/base, 12 for large) per I-JEPA guidance. Not independently swept.
- **dropout**: 0.2 for small (more regularization for fewer params), 0.1 for base/large. Fixed per size.

### Dataset-Specific Notes

| Dataset | Users  | Txns  | MCC emb | Temporal avail | Local target |
|---------|--------|-------|---------|----------------|--------------|
| Churn   | 5K     | 490K  | 24      | yes            | churn_target |
| Default | 7K     | 2M    | 16      | yes            | default_target |
| HSBC    | 4K     | 234K  | 24      | yes            | churn_target |
| Age     | 30K    | 26M   | 16      | no (serial day)| none         |

---

## 3. Code Review: Issues Found

### Issue 1: `use_temporal_encoding` CLI flag sets `True` or `None`, not `True/False`

In `train_jepa.py`:
```python
'use_temporal_encoding': True if args.use_temporal_encoding else None,
```

The `action="store_true"` argparse flag means `args.use_temporal_encoding` is `False` when not passed. But you map `False → None`, and `None` doesn't override the config default. This means if the config default is `False` and you don't pass `--use-temporal-encoding`, it stays `False` correctly. But if the config default were `True`, you'd have no CLI way to force it to `False`. This is fragile.

**Fix**: Add a `--no-temporal-encoding` flag (like you did for cosine annealing), or switch to Hydra where `use_temporal_encoding=false` is a direct override.

### Issue 2: Predictor `nhead` hard-coupled to encoder `nhead`

In `Predictor.__init__`, the predictor transformer uses `nhead=nhead` where `nhead` is the encoder's head count. For `small` (nhead=4), `predictor_d_model=96` → 96/4=24 per head, which is fine. For `base` (nhead=8), `predictor_d_model=192` → 192/8=24, also fine. But this coupling is implicit — if someone changes nhead without adjusting predictor_d_model, they'll get a crash or suboptimal head dimensions.

**Fix**: Either make `predictor_nhead` a separate config parameter, or add an assertion: `assert predictor_d_model % nhead == 0`.

### Issue 3: `sample_masks` has a Python loop over batch dimension

```python
context_indices_list = []
for b in range(batch_size):
    pos = ctx_valid_mask[b].nonzero(as_tuple=False).squeeze(1)
    context_indices_list.append(pos[:min_ctx])
context_indices = torch.stack(context_indices_list, dim=0)
```

This is a per-sample Python loop. For batch_size=128, it's probably fine (microseconds per iteration), but it's not vectorized. This isn't a critical issue but could be replaced with `torch.topk` on the mask for a fully vectorized version.

**Verdict**: Low priority. Only worth fixing if profiling shows it's a bottleneck.

### Issue 4: `diagnostics.py` uses `model.get_embedding` for context encoder

```python
ctx = model.get_embedding(mcc, amount, time_bucket=time_bucket, intra_day_rank=intra_day_rank)
```

But `get_embedding` routes through the **target encoder**, not the context/online encoder. So `ctx_embs` and `tgt_embs` are actually both from the target encoder. The encoder cosine similarity metric is therefore always ~1.0 and meaningless.

**Fix**: Add a `get_online_embedding` method to JEPA, or directly call `model.online_encoder(...)` in diagnostics.

### Issue 5: Validation loss uses masking (calls `model.forward`)

The `validate()` function calls `model(mcc, amount, ...)` which runs the full JEPA forward pass including random mask sampling. This means validation loss has stochastic noise from different masks each time. It's not wrong per se (the paper likely does the same), but it means val loss curves will be noisier than necessary.

**Verdict**: Acceptable for now. If you want smoother val curves, you could fix the mask random seed during validation.

### Issue 6: No seed control for reproducibility

`train_jepa.py` doesn't set `torch.manual_seed`, `np.random.seed`, or `random.seed`. The mask sampling uses `random.uniform` and `torch.randint`, and the data split uses `np.random.default_rng(42)`. Without seeding, two runs with identical configs won't produce identical results.

**Fix**: Add at the top of `main()`:
```python
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

### Issue 7: `download_data.py` has wrong path for churn

```python
'churn': Path('/Users/arnaldpaguio/Documents/Portofolio /LatentLedger/data/age.parquet'),
```

The churn entry points to `age.parquet`. This is a copy-paste error.

**Fix**: Change to `data/churn.parquet`.

### Issue 8: VICReg code is dead weight

VICReg weights default to 0.0 everywhere, but the code path and CLI flags still exist. Since you're removing VICReg from the experiment matrix, strip it from the model's forward pass, the training loop, and the config. Less code = fewer bugs.

---

## 4. Age Dataset Performance: CPU Bottleneck Analysis

### The Problem

Age has 26M transactions across 30K users. Your preprocessing does:

1. `pd.read_parquet(path)` — reads the full 26M-row DataFrame
2. `df.sort_values(["user_id", "timestamp"])` — sorts 26M rows
3. `df.groupby("user_id", sort=False)` — groups into 30K groups
4. **For each of 30K users**: `grp.sort_values("timestamp")`, `compute_temporal_features(grp["timestamp"])`, MCC encoding via list comprehension, building a dict of Python lists

Step 4 is where the time goes. You're iterating over 30K groups in pure Python, and for each group you're:
- Calling `pd.to_datetime()` again inside `compute_temporal_features` (redundant if already datetime)
- Doing `ts.groupby(calendar_day).cumcount()` — a pandas groupby inside a loop
- Converting everything to Python lists with `.tolist()`

For Age, the median sequence length is 863 transactions, so the inner loop processes ~26M elements one group at a time. This is fundamentally a CPU-bound Python operation.

### Diagnosis Steps (to confirm)

Add timing instrumentation:

```python
import time

t0 = time.time()
df = pd.read_parquet(path)
print(f"Read parquet: {time.time()-t0:.1f}s")

t1 = time.time()
df = df.sort_values(["user_id", "timestamp"])
print(f"Sort: {time.time()-t1:.1f}s")

t2 = time.time()
grouped = df.groupby("user_id", sort=False)
records = []
for user_id, grp in grouped:
    # ... existing preprocessing ...
    records.append(rec)
print(f"Group+process: {time.time()-t2:.1f}s ({len(records)} users)")
```

I expect you'll find that the group+process loop takes 80%+ of the total time.

### Fixes (in order of impact)

**Fix 1: Cache preprocessed records to disk (biggest win)**

Preprocess once, save as pickle/msgpack, reload on subsequent runs. This turns a 5-10 minute preprocessing step into a 5-second load:

```python
import pickle
cache_path = parquet_path.with_suffix(f'.{dataset}_cache.pkl')
if cache_path.exists() and cache_path.stat().st_mtime > parquet_path.stat().st_mtime:
    with open(cache_path, 'rb') as f:
        return pickle.load(f)
# ... do full preprocessing ...
with open(cache_path, 'wb') as f:
    pickle.dump(records, f)
```

**Fix 2: Vectorize temporal features computation**

Instead of calling `compute_temporal_features` per user inside the loop, compute time_bucket and intra_day_rank for the entire DataFrame at once before grouping:

```python
df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
df["calendar_day"] = df["timestamp_dt"].dt.normalize()

# Per-user first day (vectorized)
first_day = df.groupby("user_id")["calendar_day"].transform("first")
df["time_bucket"] = (df["calendar_day"] - first_day).dt.days.clip(0, 1023)

# Intra-day rank (vectorized)
df["intra_day_rank"] = df.groupby(["user_id", "calendar_day"]).cumcount().clip(upper=31)
```

This replaces 30K `compute_temporal_features` calls with 2 vectorized pandas operations on the full 26M-row DataFrame, which is dramatically faster.

**Fix 3: Use numpy arrays in records instead of Python lists**

Replace `.tolist()` with keeping numpy arrays. The `TransactionDataset.__getitem__` already converts to tensors, and `torch.tensor(numpy_array)` is faster than `torch.tensor(python_list)`.

**Fix 4: Use `num_workers > 0` in DataLoader**

Your current `get_dataloaders` uses `num_workers=0`. For Age with 30K users, the DataLoader's `__getitem__` is called sequentially. Setting `num_workers=4` (or more on Colab) will parallelize the per-sample tensor construction.

### Expected Improvement

With Fix 1 (caching), subsequent runs skip preprocessing entirely. With Fix 2, even the first run should drop from minutes to seconds for the preprocessing step. Together, these should make Age dataset loading comparable to the smaller datasets.

---

## Summary: What to Give Cursor

Here's the ordered task list for implementation (**Hydra-first** — do the config/sweep infrastructure before the smaller cleanups so new runs and W&B groups are correct from the start):

1. **Add Hydra config structure** and migrate the training entrypoint (`@hydra.main`, `configs/` as in §1, optional `scripts/train_jepa_hydra.py`). Wire **structured W&B init**, **run naming / group / tags**, **`cfg.seed`** with full `torch` / `numpy` / `random` (and CUDA) seeding (Issue 6), and **`use_temporal_encoding` as a direct boolean override** with auto-fallback when `not cfg.dataset.has_timestamps` (Issue 1). Deprecate or remove the fragile argparse temporal flag once Hydra is the primary path.
2. **Mirror validation** under Hydra if you split scripts (`validate_jepa_hydra.py` or shared `main` with mode) so the multirun commands in §1 stay valid end-to-end.
3. **Fix `download_data.py`** churn path (Issue 7) — trivial, independent.
4. **Remove VICReg** from model, training loop, and config (Issue 8).
5. **Fix diagnostics** context encoder measurement (Issue 4).
6. **Add preprocessing cache** for Age dataset (Fix 1).
7. **Vectorize temporal features** in preprocessing.py (Fix 2).
8. **Add predictor nhead assertion** (Issue 2) or explicit `predictor_nhead` in config.

**Risk note**: Item 1 touches the training entrypoint and how every run is launched; smoke-test on Churn (single run + one `--multirun` slice) before relying on full-grid sweeps. Items 3–8 are mostly independent and can proceed in parallel after Hydra is merged, except that VICReg removal (4) should align with whatever config keys Hydra exposes.