# Jane Street Real-Time Market Data Forecasting

A modeling pipeline for the [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) Kaggle competition. The goal is to predict `responder_6` — a financial return signal — from 79 anonymized market features across 39 symbols.

> **Note:** This was completed after the competition closed, as a portfolio and learning project.

---

## Problem Overview

- **Target:** `responder_6` — a bounded continuous return signal in [-5, 5]
- **Features:** 79 anonymized features per row; 3 are categorical (`feature_09`, `10`, `11`)
- **Data:** Time-series market data partitioned by date, ~39 symbols, ~1600 date IDs
- **Metric:** Weighted zero-mean R² (weights are provided per row)

---

## Key EDA Findings

- Raw feature correlations with `responder_6` are tiny (max ~0.07) — signal is weak and non-linear
- `feature_06` and `feature_07` are the most predictive raw features
- Heavy NaNs before `date_id = 247` — data before this cutoff is dropped
- Weights are dominated by symbols 1, 13, and 19 — model performance on these matters most
- `responder_6` is fat-tailed, near-zero mean, and shows near-zero autocorrelation at all lags

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | Exploratory data analysis — target distribution, NaN patterns, weight analysis, feature correlations, responder correlations |
| `02_lgbm_baseline.ipynb` | LGBM global model with market averages and rolling features; symbol-specific residual modeling for problem symbols (4, 12, 17, 28) |
| `03_factor_model.ipynb` | PCA factor model (10/20/30 components) + LGBM on factors; generalization test on partition 6 |
| `scratchpad.ipynb` | Early exploratory attempts — not intended to run end-to-end |

---

## Modeling Approach

### Preprocessing
- Drop `date_id < 247` (excessive NaNs in early data)
- Drop categorical features (`feature_09`, `feature_10`, `feature_11`)
- Fill remaining NaNs with zero
- Reduce memory via float32 casting

### Feature Engineering
- Top 30 correlated features selected per fold
- Market averages: `groupby(['date_id', 'time_id'])` mean for cross-sectional signal
- Rolling statistics (window=1000) on top 10 features per symbol

### Models

**Baseline — LGBM (`02_lgbm_baseline.ipynb`)**
- Global LightGBM on engineered features
- Symbol-specific residual models for high-variance symbols
- Global residual stacking (LGBM → residuals → second LGBM)

**Factor Model — PCA + LGBM (`03_factor_model.ipynb`)**
- PCA compression of feature space (10, 20, 30 components tested)
- LGBM on PCA factors — 20 components gave best generalization
- Ridge regression on factors as linear baseline

---

## Results

| Model | Weighted R² (val) |
|---|---|
| LGBM baseline | ~0.003 |
| LGBM + residual stacking | ~0.003 |
| PCA (20) + LGBM | ~0.004 |

> R² values are low by design — this is a notoriously noisy financial prediction task. The competition leaderboard scores are in a similar range.

---

## Repository Structure

```
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_lgbm_baseline.ipynb
│   ├── 03_factor_model.ipynb
│   └── scratchpad.ipynb
├── README.md
└── requirements.txt
```

---

## Data

Data is available on Kaggle and requires a Kaggle account to download:
```bash
kaggle competitions download -c jane-street-real-time-market-data-forecasting
```
All notebooks use Kaggle paths (`/kaggle/input/...`) and are designed to run in a Kaggle kernel environment.

---

## Requirements

```bash
pip install -r requirements.txt
```

See `requirements.txt` for full list.

---

## Author

**Akshay Sakanaveeti**  
Department of Statistics and Operations Research  
University of North Carolina at Chapel Hill  
sakshay@unc.edu
