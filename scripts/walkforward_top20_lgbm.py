from pathlib import Path
import gc
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "top20_p8"
MODEL_DIR = PROJECT_ROOT / "models" / "lgbm" / "top20_walkforward"
LOG_DIR = PROJECT_ROOT / "logs"

TRAIN_PATH = DATA_DIR / "train.parquet"
VALID_PATH = DATA_DIR / "valid.parquet"
REPORT_PATH = LOG_DIR / "walkforward_top20_lgbm_report.json"
RESULTS_PATH = MODEL_DIR / "walkforward_results.csv"

TARGET_COL = "responder_6"
WEIGHT_COL = "weight"
DATE_COL = "date_id"
CATEGORICAL_COLS = ["symbol_id", "feature_11"]
SEED = 42
BLOCK_DAYS = 5


def weighted_r2(y_true, y_pred, weight):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    numerator = np.sum(weight * np.square(y_true - y_pred))
    denominator = np.sum(weight * np.square(y_true))
    if denominator == 0:
        return np.nan
    return 1.0 - numerator / denominator


def load_data():
    if not TRAIN_PATH.exists() or not VALID_PATH.exists():
        raise FileNotFoundError(f"Missing train/valid parquet in {DATA_DIR}")
    train_df = pd.read_parquet(TRAIN_PATH, engine="pyarrow")
    valid_df = pd.read_parquet(VALID_PATH, engine="pyarrow")
    df = pd.concat([train_df, valid_df], ignore_index=True)
    df = df.sort_values([DATE_COL, "time_id", "symbol_id"]).reset_index(drop=True)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    print(f"combined_shape={df.shape}", flush=True)
    return df


def get_feature_cols(df):
    return [col for col in df.columns if col not in [DATE_COL, WEIGHT_COL, TARGET_COL]]


def make_dataset(df, feature_cols, categorical_cols):
    return lgb.Dataset(
        df[feature_cols],
        label=df[TARGET_COL],
        weight=df[WEIGHT_COL],
        feature_name=feature_cols,
        categorical_feature=[col for col in categorical_cols if col in feature_cols],
        free_raw_data=False,
    )


def params():
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "device_type": "gpu",
        "gpu_use_dp": False,
        "max_bin": 63,
        "num_leaves": 64,
        "learning_rate": 0.03,
        "min_data_in_leaf": 1000,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 5.0,
        "seed": SEED,
        "feature_pre_filter": False,
        "verbosity": -1,
    }


def fold_starts(valid_dates):
    return list(valid_dates[::BLOCK_DAYS])


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    feature_cols = get_feature_cols(df)
    categorical_cols = [col for col in CATEGORICAL_COLS if col in feature_cols]
    valid_dates = np.array(sorted(pd.read_parquet(VALID_PATH, columns=[DATE_COL])[DATE_COL].unique()))
    starts = fold_starts(valid_dates)

    results = []
    for fold_idx, start_date in enumerate(starts, start=1):
        end_date = min(start_date + BLOCK_DAYS - 1, valid_dates[-1])
        train_df = df[df[DATE_COL] < start_date]
        valid_df = df[(df[DATE_COL] >= start_date) & (df[DATE_COL] <= end_date)]
        if len(train_df) == 0 or len(valid_df) == 0:
            continue

        print(
            f"fold={fold_idx} valid_dates={int(start_date)}-{int(end_date)} "
            f"train_rows={len(train_df):,} valid_rows={len(valid_df):,}",
            flush=True,
        )
        train_set = make_dataset(train_df, feature_cols, categorical_cols)
        valid_set = make_dataset(valid_df, feature_cols, categorical_cols)
        start = time.time()
        model = lgb.train(
            params(),
            train_set,
            valid_sets=[train_set, valid_set],
            valid_names=["train", "valid"],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )
        elapsed = time.time() - start
        preds = model.predict(valid_df[feature_cols], num_iteration=model.best_iteration)
        score = weighted_r2(valid_df[TARGET_COL], preds, valid_df[WEIGHT_COL])
        model_path = MODEL_DIR / f"fold_{fold_idx:02d}_{int(start_date)}_{int(end_date)}.txt"
        model.save_model(model_path)

        result = {
            "fold": fold_idx,
            "valid_date_min": int(start_date),
            "valid_date_max": int(end_date),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "best_iteration": int(model.best_iteration),
            "valid_weighted_r2": float(score),
            "elapsed_seconds": float(elapsed),
            "model_path": str(model_path),
        }
        results.append(result)
        print(f"fold_result={result}", flush=True)
        del train_set, valid_set, model
        gc.collect()

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    report = {
        "train_path": str(TRAIN_PATH),
        "valid_path": str(VALID_PATH),
        "block_days": BLOCK_DAYS,
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "results": results,
        "mean_valid_weighted_r2": float(results_df["valid_weighted_r2"].mean()),
        "results_path": str(RESULTS_PATH),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("DONE", flush=True)
    print(f"report={REPORT_PATH}", flush=True)
    print(f"results={RESULTS_PATH}", flush=True)
    print(f"mean_valid_weighted_r2={report['mean_valid_weighted_r2']:.8f}", flush=True)


if __name__ == "__main__":
    main()
