# Jane Street Real-Time Market Data Forecasting

Project workspace for the Jane Street Real-Time Market Data Forecasting Kaggle competition.

Current focus: partition 8 only, memory-aware EDA/cleaning, and Longleaf GPU baselines.

## Current Workflow

| File | Purpose |
|---|---|
| `notebooks/01_kaggle_eda.ipynb` | Original Kaggle EDA notebook |
| `notebooks/02_kaggle_lgbm.ipynb` | Original Kaggle LightGBM experiments |
| `notebooks/03_kaggle_factor.ipynb` | Original Kaggle factor/PCA experiments |
| `notebooks/01_loader_p8.ipynb` | Batch parquet loader for partition 8 without loading the full dataset into memory |
| `notebooks/02_sample_eda_p8.ipynb` | Small sample EDA for missingness, weights, symbols, and feature relationships |
| `notebooks/03_stream_eda_p8.ipynb` | Streaming EDA over full partition 8 using aggregates instead of full in-memory data |
| `notebooks/04_clean_p8.ipynb` | Notebook version of partition 8 cleaning by dropping missing rows |
| `notebooks/05_lgbm_sample_p8.ipynb` | Small-sample LightGBM baseline for checking setup and top features |
| `notebooks/06_top20_lgbm_eval.ipynb` | Post-training evaluation plots for top-20 LightGBM on train/valid/test |
| `scripts/clean_p8.py` | Longleaf script to clean partition 8 in batches |
| `scripts/validate_p8.py` | Longleaf script to validate cleaned partition 8 |
| `scripts/gpu_check.py` | Longleaf script to test LightGBM, XGBoost, and PyTorch GPU availability |
| `scripts/lgbm_top20.py` | Longleaf LightGBM baseline and top-20 feature importance script |
| `scripts/make_top20_p8.py` | Creates reduced top-20 train/valid/test parquet files |
| `scripts/train_top20_lgbm.py` | Tunes LightGBM on top-20 train/valid parquet files |

## Data

Raw and cleaned parquet files are not tracked in git.

Expected Longleaf paths:

```text
/users/s/a/sakshay/js_kaggle/data/part_8.parquet
/users/s/a/sakshay/js_kaggle/data/clean/partition_8_drop_missing/part-0.parquet
```

Cleaning result from partition 8:

```text
rows_read=6,140,024
rows_written=5,538,291
rows_dropped=601,733
drop_rate=0.098002
```

## Modeling

The first baseline is LightGBM on cleaned partition 8:

- target: `responder_6`
- weight: `weight`
- features: all `feature_` columns plus `symbol_id` and `time_id` if present
- validation: last 20 `date_id` values
- output: feature importance and top 20 selected features

## Longleaf Files

Slurm submit files:

```text
clean_p8.sbatch
validate_p8.sbatch
gpu_check.sbatch
gpu_check_a100.sbatch
lgbm_top20.sbatch
make_top20_p8.sbatch
train_top20_lgbm.sbatch
```

Cluster-specific Python dependencies are listed in `requirements-longleaf.txt`.

## Repository Notes

Ignored local paths include:

```text
data/
models/
logs/
experiments/
archive/
*.parquet
```

Old exploratory notebooks are kept locally in `archive/old_notebooks/` and are not intended for GitHub.
