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
| `notebooks/07_tree_phase1_p8.ipynb` | Phase 1 tree-model checks for LGBM, XGBoost, and walk-forward validation |
| `notebooks/08_phase1_model_compare.ipynb` | Phase 1 model comparison dashboard for LGBM, XGBoost, and walk-forward results |
| `notebooks/09_rolling_features_p8.ipynb` | Phase 2 rolling-feature check on sampled date windows |
| `scripts/clean_p8.py` | Longleaf script to clean partition 8 in batches |
| `scripts/validate_p8.py` | Longleaf script to validate cleaned partition 8 |
| `scripts/gpu_check.py` | Longleaf script to test LightGBM, XGBoost, and PyTorch GPU availability |
| `scripts/lgbm_top20.py` | Longleaf LightGBM baseline and top-20 feature importance script |
| `scripts/make_top20_p8.py` | Creates reduced top-20 train/valid/test parquet files |
| `scripts/train_top20_lgbm.py` | Tunes LightGBM on top-20 train/valid parquet files |
| `scripts/train_top20_xgb.py` | Tunes XGBoost on top-20 train/valid parquet files |
| `scripts/walkforward_top20_lgbm.py` | Walk-forward LightGBM validation over validation dates |
| `scripts/make_rolling_p8.py` | Creates top-20 plus rolling-feature train/valid/test parquet files |
| `scripts/train_rolling_lgbm.py` | Tunes LightGBM on the rolling-feature dataset |
| `scripts/train_rolling_xgb.py` | Tunes XGBoost on the rolling-feature dataset |

Model artifacts:

```text
models/lgbm/top20_lgbm/
models/lgbm/top20_walkforward/
models/lgbm/top20_rolling_lgbm/
models/xgboost/top20_xgb/
models/xgboost/top20_rolling_xgb/
```

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

## Current Results

Validation weighted R2 on partition 8:

| Model | Feature set | Valid weighted R2 |
|---|---:|---:|
| XGBoost | top 20 + rolling | 0.005567 |
| XGBoost | top 20 | 0.005500 |
| LightGBM | top 20 + rolling | 0.004498 |
| LightGBM | top 20 | 0.004264 |

Rolling features helped slightly, with XGBoost still the strongest tree model so far.

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
train_top20_xgb.sbatch
walkforward_top20_lgbm.sbatch
make_rolling_p8.sbatch
train_rolling_lgbm.sbatch
train_rolling_xgb.sbatch
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
