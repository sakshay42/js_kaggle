from pathlib import Path
import gc
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "top20_rolling_p8"
MODEL_DIR = PROJECT_ROOT / "models" / "xgboost" / "top20_rolling_xgb"
LOG_DIR = PROJECT_ROOT / "logs"

TRAIN_PATH = DATA_DIR / "train.parquet"
VALID_PATH = DATA_DIR / "valid.parquet"
REPORT_PATH = LOG_DIR / "train_rolling_xgb_report.json"
TUNING_PATH = MODEL_DIR / "tuning_results.csv"
BEST_MODEL_PATH = MODEL_DIR / "best_xgb.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "best_feature_importance.csv"

TARGET_COL = "responder_6"
WEIGHT_COL = "weight"
DATE_COL = "date_id"
SEED = 42


PARAM_GRID = [
    {
        "name": "baseline",
        "max_depth": 5,
        "learning_rate": 0.03,
        "min_child_weight": 100,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
    },
    {
        "name": "shallower",
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 100,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
    },
    {
        "name": "more_regularized",
        "max_depth": 5,
        "learning_rate": 0.03,
        "min_child_weight": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
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
    print(f"reading_{name}={path}", flush=True)
    df = pd.read_parquet(path, engine="pyarrow")
    print(f"{name}_shape={df.shape}", flush=True)
    return df


def get_feature_cols(df):
    return [col for col in df.columns if col not in [DATE_COL, WEIGHT_COL, TARGET_COL]]


def xgb_feature_frame(df, feature_cols):
    X = df[feature_cols].copy()
    for col in X.columns:
        if str(X[col].dtype) == "category":
            X[col] = X[col].cat.codes.astype("int16")
    return X


def make_dmatrix(df, feature_cols):
    return xgb.DMatrix(
        data=xgb_feature_frame(df, feature_cols),
        label=df[TARGET_COL],
        weight=df[WEIGHT_COL],
        feature_names=feature_cols,
    )


def base_params():
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda",
        "seed": SEED,
        "verbosity": 1,
    }


def train_one(config, train_dm, valid_dm, valid_df):
    params = base_params()
    params.update({key: value for key, value in config.items() if key != "name"})

    print(f"training_config={config['name']} params={params}", flush=True)
    start = time.time()
    model = xgb.train(
        params=params,
        dtrain=train_dm,
        num_boost_round=2000,
        evals=[(train_dm, "train"), (valid_dm, "valid")],
        early_stopping_rounds=100,
        verbose_eval=100,
    )
    elapsed = time.time() - start

    preds = model.predict(valid_dm, iteration_range=(0, model.best_iteration + 1))
    score = weighted_r2(valid_df[TARGET_COL], preds, valid_df[WEIGHT_COL])
    result = {
        "name": config["name"],
        "valid_weighted_r2": float(score),
        "best_iteration": int(model.best_iteration),
        "best_score_rmse": float(model.best_score),
        "elapsed_seconds": float(elapsed),
        **{key: value for key, value in config.items() if key != "name"},
    }
    print(
        f"done_config={config['name']} "
        f"best_iteration={model.best_iteration} "
        f"valid_weighted_r2={score:.8f} "
        f"elapsed_seconds={elapsed:.1f}",
        flush=True,
    )
    return model, result


def feature_importance_frame(model, feature_cols):
    gain = model.get_score(importance_type="gain")
    weight = model.get_score(importance_type="weight")
    rows = []
    for feature in feature_cols:
        rows.append(
            {
                "feature": feature,
                "importance_gain": float(gain.get(feature, 0.0)),
                "importance_split": float(weight.get(feature, 0.0)),
            }
        )
    return (
        pd.DataFrame(rows)
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
    feature_cols = get_feature_cols(train_df)
    print(f"feature_count={len(feature_cols)}", flush=True)
    print(f"features={feature_cols}", flush=True)

    train_dm = make_dmatrix(train_df, feature_cols)
    valid_dm = make_dmatrix(valid_df, feature_cols)

    results = []
    best_model = None
    best_result = None
    for config in PARAM_GRID:
        model, result = train_one(config, train_dm, valid_dm, valid_df)
        results.append(result)
        if best_result is None or result["valid_weighted_r2"] > best_result["valid_weighted_r2"]:
            best_model = model
            best_result = result
        gc.collect()

    tuning = pd.DataFrame(results).sort_values("valid_weighted_r2", ascending=False)
    tuning.to_csv(TUNING_PATH, index=False)

    best_model.save_model(BEST_MODEL_PATH)
    importance = feature_importance_frame(best_model, feature_cols)
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    report = {
        "data_dir": str(DATA_DIR),
        "train": split_summary(train_df),
        "valid": split_summary(valid_df),
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "best": best_result,
        "tuning_results": results,
        "tuning_path": str(TUNING_PATH),
        "best_model_path": str(BEST_MODEL_PATH),
        "feature_importance_path": str(FEATURE_IMPORTANCE_PATH),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("DONE", flush=True)
    print(f"report={REPORT_PATH}", flush=True)
    print(f"tuning={TUNING_PATH}", flush=True)
    print(f"best_model={BEST_MODEL_PATH}", flush=True)
    print(f"best={best_result}", flush=True)


if __name__ == "__main__":
    main()
