from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "clean" / "partition_8_drop_missing" / "part-0.parquet"
MODEL_DIR = PROJECT_ROOT / "models" / "lgbm_baseline_top20"
LOG_DIR = PROJECT_ROOT / "logs"

TARGET_COL = "responder_6"
WEIGHT_COL = "weight"
DATE_COL = "date_id"
EXTRA_FEATURE_COLS = ["symbol_id", "time_id"]
KNOWN_CATEGORICAL_COLS = ["symbol_id", "feature_09", "feature_10", "feature_11"]
VAL_DATE_COUNT = 20
TOP_N = 20
SEED = 42


def weighted_r2(y_true, y_pred, weight):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    numerator = np.sum(weight * np.square(y_true - y_pred))
    denominator = np.sum(weight * np.square(y_true))
    if denominator == 0:
        return np.nan
    return 1.0 - numerator / denominator


def json_value(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing cleaned parquet file: {DATA_PATH}")

    print(f"reading={DATA_PATH}", flush=True)
    df = pd.read_parquet(DATA_PATH, engine="pyarrow")
    print(f"rows={len(df):,} columns={len(df.columns):,}", flush=True)
    return df


def get_columns(df):
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    extra_cols = [col for col in EXTRA_FEATURE_COLS if col in df.columns]
    model_cols = feature_cols + extra_cols

    missing = [col for col in [TARGET_COL, WEIGHT_COL, DATE_COL] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not feature_cols:
        raise ValueError("No feature_ columns found")

    categorical_cols = [col for col in KNOWN_CATEGORICAL_COLS if col in model_cols]
    return feature_cols, model_cols, categorical_cols


def time_split(df):
    unique_dates = np.array(sorted(df[DATE_COL].dropna().unique()))
    if len(unique_dates) <= VAL_DATE_COUNT:
        raise ValueError(f"Need more than {VAL_DATE_COUNT} unique dates for validation split")

    val_dates = unique_dates[-VAL_DATE_COUNT:]
    train_mask = df[DATE_COL] < val_dates[0]
    val_mask = df[DATE_COL] >= val_dates[0]

    split_info = {
        "train_date_min": json_value(df.loc[train_mask, DATE_COL].min()),
        "train_date_max": json_value(df.loc[train_mask, DATE_COL].max()),
        "val_date_min": json_value(df.loc[val_mask, DATE_COL].min()),
        "val_date_max": json_value(df.loc[val_mask, DATE_COL].max()),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "unique_dates": int(len(unique_dates)),
        "val_date_count": int(len(val_dates)),
    }
    return train_mask, val_mask, split_info


def cast_categoricals(df, categorical_cols):
    for col in categorical_cols:
        df[col] = df[col].astype("category")


def make_dataset(df, rows, cols, categorical_cols):
    return lgb.Dataset(
        df.loc[rows, cols],
        label=df.loc[rows, TARGET_COL],
        weight=df.loc[rows, WEIGHT_COL],
        feature_name=cols,
        categorical_feature=[col for col in categorical_cols if col in cols],
        free_raw_data=False,
    )


def train_model(df, train_mask, val_mask, cols, categorical_cols, model_name):
    train_set = make_dataset(df, train_mask, cols, categorical_cols)
    val_set = make_dataset(df, val_mask, cols, categorical_cols)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "device_type": "gpu",
        "gpu_use_dp": False,
        "max_bin": 63,
        "num_leaves": 64,
        "learning_rate": 0.03,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "min_data_in_leaf": 500,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "seed": SEED,
        "feature_pre_filter": False,
        "verbosity": -1,
    }

    print(f"training={model_name} features={len(cols):,}", flush=True)
    model = lgb.train(
        params,
        train_set,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50),
        ],
    )

    preds = model.predict(df.loc[val_mask, cols], num_iteration=model.best_iteration)
    score = weighted_r2(df.loc[val_mask, TARGET_COL], preds, df.loc[val_mask, WEIGHT_COL])
    print(f"{model_name}_best_iteration={model.best_iteration}", flush=True)
    print(f"{model_name}_valid_weighted_r2={score:.8f}", flush=True)
    return model, score


def feature_importance_frame(model):
    return (
        pd.DataFrame(
            {
                "feature": model.feature_name(),
                "importance_gain": model.feature_importance(importance_type="gain"),
                "importance_split": model.feature_importance(importance_type="split"),
            }
        )
        .sort_values(["importance_gain", "importance_split"], ascending=False)
        .reset_index(drop=True)
    )


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    feature_cols, model_cols, categorical_cols = get_columns(df)
    train_mask, val_mask, split_info = time_split(df)
    cast_categoricals(df, categorical_cols)

    all_model, all_score = train_model(
        df=df,
        train_mask=train_mask,
        val_mask=val_mask,
        cols=model_cols,
        categorical_cols=categorical_cols,
        model_name="all_features",
    )
    all_model_path = MODEL_DIR / "lgbm_all_features.txt"
    all_model.save_model(all_model_path)

    importance = feature_importance_frame(all_model)
    importance_path = MODEL_DIR / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    top20_features = importance.head(TOP_N)["feature"].tolist()
    top20_path = MODEL_DIR / "top20_features.txt"
    top20_path.write_text("\n".join(top20_features) + "\n")

    top20_df = df[[DATE_COL, WEIGHT_COL, TARGET_COL] + top20_features].copy()
    top20_categorical_cols = [col for col in categorical_cols if col in top20_features]
    print(f"top20_dataset_shape={top20_df.shape}", flush=True)
    top20_model, top20_score = train_model(
        df=top20_df,
        train_mask=train_mask,
        val_mask=val_mask,
        cols=top20_features,
        categorical_cols=top20_categorical_cols,
        model_name="top20_features",
    )
    top20_model_path = MODEL_DIR / "lgbm_top20_features.txt"
    top20_model.save_model(top20_model_path)

    report = {
        "data_path": str(DATA_PATH),
        "rows": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "model_feature_count": int(len(model_cols)),
        "categorical_cols": categorical_cols,
        "split": split_info,
        "all_features": {
            "best_iteration": int(all_model.best_iteration),
            "valid_weighted_r2": float(all_score),
            "model_path": str(all_model_path),
        },
        "top20_features": {
            "features": top20_features,
            "source": "all_features_lgbm_gain_importance",
            "categorical_cols": top20_categorical_cols,
            "dataset_shape": [int(top20_df.shape[0]), int(top20_df.shape[1])],
            "best_iteration": int(top20_model.best_iteration),
            "valid_weighted_r2": float(top20_score),
            "model_path": str(top20_model_path),
        },
        "importance_path": str(importance_path),
        "top20_path": str(top20_path),
    }

    report_path = LOG_DIR / "lgbm_baseline_top20_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("DONE", flush=True)
    print(f"report={report_path}", flush=True)
    print(f"importance={importance_path}", flush=True)
    print(f"top20={top20_path}", flush=True)
    print("top20_features:", flush=True)
    for feature in top20_features:
        print(feature, flush=True)


if __name__ == "__main__":
    main()
