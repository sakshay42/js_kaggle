from pathlib import Path
import gc
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "top20_rolling_p8"
MODEL_DIR = PROJECT_ROOT / "models" / "lgbm" / "top20_rolling_lgbm"
LOG_DIR = PROJECT_ROOT / "logs"

TRAIN_PATH = DATA_DIR / "train.parquet"
VALID_PATH = DATA_DIR / "valid.parquet"
REPORT_PATH = LOG_DIR / "train_rolling_lgbm_report.json"
TUNING_PATH = MODEL_DIR / "tuning_results.csv"
BEST_MODEL_PATH = MODEL_DIR / "best_lgbm.txt"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "best_feature_importance.csv"

TARGET_COL = "responder_6"
WEIGHT_COL = "weight"
DATE_COL = "date_id"
CATEGORICAL_COLS = ["symbol_id", "feature_11"]
SEED = 42

PARAM_GRID = [
    {
        "name": "baseline",
        "num_leaves": 64,
        "learning_rate": 0.03,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "lambda_l2": 1.0,
    },
    {
        "name": "more_regularized",
        "num_leaves": 64,
        "learning_rate": 0.03,
        "min_data_in_leaf": 1000,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "lambda_l2": 5.0,
    },
    {
        "name": "smaller_leaves",
        "num_leaves": 31,
        "learning_rate": 0.03,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "lambda_l2": 1.0,
    },
]


def weighted_r2(y_true, y_pred, weight):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    numerator = np.sum(weight * np.square(y_true - y_pred))
    denominator = np.sum(weight * np.square(y_true))
    return np.nan if denominator == 0 else 1.0 - numerator / denominator


def load_split(path, name):
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} parquet: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    print(f"{name}_shape={df.shape}", flush=True)
    return df


def feature_cols(df):
    return [col for col in df.columns if col not in [DATE_COL, WEIGHT_COL, TARGET_COL]]


def cast_categoricals(train_df, valid_df, categorical_cols):
    for col in categorical_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype("category")
            valid_df[col] = valid_df[col].astype("category")


def make_dataset(df, features, categorical_cols):
    return lgb.Dataset(
        df[features],
        label=df[TARGET_COL],
        weight=df[WEIGHT_COL],
        feature_name=features,
        categorical_feature=[col for col in categorical_cols if col in features],
        free_raw_data=False,
    )


def base_params():
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "device_type": "gpu",
        "gpu_use_dp": False,
        "max_bin": 63,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "seed": SEED,
        "feature_pre_filter": False,
        "verbosity": -1,
    }


def train_one(config, train_set, valid_set, valid_df, features):
    params = base_params()
    params.update({k: v for k, v in config.items() if k != "name"})
    print(f"training_config={config['name']} params={params}", flush=True)
    start = time.time()
    model = lgb.train(
        params,
        train_set,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    elapsed = time.time() - start
    pred = model.predict(valid_df[features], num_iteration=model.best_iteration)
    score = weighted_r2(valid_df[TARGET_COL], pred, valid_df[WEIGHT_COL])
    result = {
        "name": config["name"],
        "valid_weighted_r2": float(score),
        "best_iteration": int(model.best_iteration),
        "elapsed_seconds": float(elapsed),
        **{k: v for k, v in config.items() if k != "name"},
    }
    print(f"done_config={config['name']} valid_weighted_r2={score:.8f}", flush=True)
    return model, result


def importance_frame(model):
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


def split_summary(df):
    return {
        "rows": int(len(df)),
        "date_min": int(df[DATE_COL].min()),
        "date_max": int(df[DATE_COL].max()),
        "unique_dates": int(df[DATE_COL].nunique()),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split(TRAIN_PATH, "train")
    valid_df = load_split(VALID_PATH, "valid")
    features = feature_cols(train_df)
    categorical_cols = [col for col in CATEGORICAL_COLS if col in features]
    cast_categoricals(train_df, valid_df, categorical_cols)
    print(f"feature_count={len(features)}", flush=True)

    train_set = make_dataset(train_df, features, categorical_cols)
    valid_set = make_dataset(valid_df, features, categorical_cols)

    results = []
    best_model = None
    best_result = None
    for config in PARAM_GRID:
        model, result = train_one(config, train_set, valid_set, valid_df, features)
        results.append(result)
        if best_result is None or result["valid_weighted_r2"] > best_result["valid_weighted_r2"]:
            best_model = model
            best_result = result
        gc.collect()

    pd.DataFrame(results).sort_values("valid_weighted_r2", ascending=False).to_csv(TUNING_PATH, index=False)
    best_model.save_model(BEST_MODEL_PATH)
    importance_frame(best_model).to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    report = {
        "data_dir": str(DATA_DIR),
        "train": split_summary(train_df),
        "valid": split_summary(valid_df),
        "feature_count": len(features),
        "categorical_cols": categorical_cols,
        "best": best_result,
        "tuning_results": results,
        "tuning_path": str(TUNING_PATH),
        "best_model_path": str(BEST_MODEL_PATH),
        "feature_importance_path": str(FEATURE_IMPORTANCE_PATH),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("DONE", flush=True)
    print(f"report={REPORT_PATH}", flush=True)
    print(f"best={best_result}", flush=True)


if __name__ == "__main__":
    main()
