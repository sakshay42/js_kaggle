# Jane Street Real-Time Market Data Forecasting

An empirical forecasting project based on the [Jane Street Real-Time Market Data Forecasting competition](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/overview). The project focuses on memory-efficient processing, chronological validation, tree models, temporal feature engineering, neural regularization, and model ensembles.

## Dataset

The analysis uses training partition 8:

- 6,140,024 raw rows and 79 anonymized `feature_` columns
- target: `responder_6`
- sample weight: `weight`
- identifiers: `date_id`, `time_id`, and `symbol_id`

Rows with missing features, target, or weight were removed in streaming batches, leaving 5,538,291 rows. The cleaned data were split chronologically:

| Split | Dates | Rows |
|---|---:|---:|
| Train | 1360-1489 | 4,162,125 |
| Validation | 1490-1509 | 680,149 |
| Test | 1510-1529 | 696,017 |

The test split is an internal out-of-time partition, not Kaggle's hidden test set.

## Approach

1. Stream parquet data with PyArrow to avoid materializing the full partition in memory.
2. Use LightGBM gain importance to reduce 81 candidate inputs to 20.
3. Tune LightGBM and XGBoost with chronological and expanding-window validation.
4. Add 5- and 20-step symbol-level rolling means and standard deviations using lagged features only.
5. Train MLP and GRU+MLP models, diagnose overfitting, and test dropout, weight decay, LayerNorm, Huber loss, early stopping, prediction shrinkage, and multiple seeds.
6. Evaluate date-by-date online neural updates and tree-neural ensembles.

Training and large experiments ran on NVIDIA GPUs through UNC Longleaf.

## Results

Weighted zero-mean R2 on the partition 8 chronological splits:

| Model | Validation R2 | Test R2 |
|---|---:|---:|
| Focused MLP + XGBoost ensemble | 0.006523 | 0.004425 |
| Focused MLP mean | 0.005984 | 0.003024 |
| Rolling XGBoost | 0.004963 | **0.004996** |
| Rolling LightGBM | 0.004498 | - |
| Original MLP | -0.003860 | -0.006744 |
| Original GRU+MLP | -0.012707 | -0.014378 |

Rolling XGBoost was the strongest later-period model. Regularization made the neural models competitive on validation, but those gains transferred less reliably to the test period.

The leading competition score was approximately `0.01`; the local score of about `0.005` is of the same order of magnitude. These values are not directly comparable because this project uses one training partition and an internal chronological split rather than the competition's hidden forecasting period.

## Repository

- `notebooks/final.ipynb`: complete project narrative and model comparison
- `notebooks/`: EDA, data engineering, feature selection, and model-analysis notebooks
- `scripts/`: cleaning, feature engineering, training, online-update, and ensemble code
- `slurm/`: Longleaf batch submissions
- `requirements.txt`: local analysis dependencies
- `requirements-longleaf.txt`: cluster training dependencies

Raw data, model binaries, logs, generated reports, and archived scratch work are excluded from Git.
