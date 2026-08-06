from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "top20_rolling_p8"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = MODEL_DIR / "ensemble" / "rolling_final"
LOG_DIR = PROJECT_ROOT / "logs"

XGB_MODEL_PATH = MODEL_DIR / "xgboost" / "top20_rolling_xgb" / "best_xgb.json"
MLP_PRED_PATH = MODEL_DIR / "neural" / "rolling_mlp" / "predictions.parquet"
GRU_PRED_PATH = MODEL_DIR / "neural" / "rolling_gru_mlp" / "predictions.parquet"
ONLINE_MLP_PRED_PATH = MODEL_DIR / "neural" / "online_rolling_mlp" / "predictions.parquet"
ONLINE_GRU_PRED_PATH = MODEL_DIR / "neural" / "online_rolling_gru_mlp" / "predictions.parquet"
REPORT_PATH = LOG_DIR / "ensemble_rolling_report.json"
WEIGHTS_PATH = OUT_DIR / "blend_weights.csv"
PRED_PATH = OUT_DIR / "predictions.parquet"

TARGET_COL = "responder_6"
WEIGHT_COL = "weight"
DATE_COL = "date_id"


def weighted_r2(y_true, y_pred, weight):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    denom = np.sum(weight * np.square(y_true))
    return np.nan if denom == 0 else 1.0 - np.sum(weight * np.square(y_true - y_pred)) / denom


def feature_cols(df):
    return [c for c in df.columns if c not in [DATE_COL, WEIGHT_COL, TARGET_COL, "_split"]]


def xgb_feature_frame(df, features):
    X = df[features].copy()
    for col in X.columns:
        if str(X[col].dtype) == "category":
            X[col] = X[col].cat.codes.astype("int16")
    return X


def add_xgb_preds(split):
    path = DATA_DIR / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path, engine="pyarrow")
    features = feature_cols(df)
    model = xgb.Booster()
    model.load_model(str(XGB_MODEL_PATH))
    dm = xgb.DMatrix(xgb_feature_frame(df, features), feature_names=features)
    return df[[DATE_COL, "time_id", "symbol_id", TARGET_COL, WEIGHT_COL]].assign(
        split=split,
        xgb_pred=model.predict(dm),
    )


def load_nn_predictions(path, pred_col):
    if not path.exists():
        print(f"missing optional predictions: {path}", flush=True)
        return None
    df = pd.read_parquet(path, engine="pyarrow")
    return df[[DATE_COL, "time_id", "symbol_id", "split", pred_col]]


def score_frame(df, pred_cols):
    rows = []
    for split, group in df.groupby("split"):
        for col in pred_cols:
            rows.append(
                {
                    "split": split,
                    "model": col.replace("_pred", ""),
                    "weighted_r2": weighted_r2(group[TARGET_COL], group[col], group[WEIGHT_COL]),
                }
            )
    return pd.DataFrame(rows)


def tune_blend(valid_df, pred_cols):
    rows = []
    best = None

    def simplex_weights(n, steps=20):
        if n == 1:
            yield [steps]
            return
        for i in range(steps + 1):
            for rest in simplex_weights(n - 1, steps - i):
                yield [i, *rest]

    for weights in simplex_weights(len(pred_cols), steps=20):
        weights = np.array(weights, dtype=np.float64) / 20.0
        pred = sum(weights[i] * valid_df[pred_cols[i]] for i in range(len(pred_cols)))
        score = weighted_r2(valid_df[TARGET_COL], pred, valid_df[WEIGHT_COL])
        row = {"weighted_r2": float(score), **{f"w_{pred_cols[i]}": float(weights[i]) for i in range(len(pred_cols))}}
        rows.append(row)
        if best is None or score > best["weighted_r2"]:
            best = row
    return pd.DataFrame(rows).sort_values("weighted_r2", ascending=False), best


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.concat([add_xgb_preds("valid"), add_xgb_preds("test")], ignore_index=True)
    keys = [DATE_COL, "time_id", "symbol_id", "split"]
    pred_df = base
    pred_cols = ["xgb_pred"]
    for path, pred_col in [
        (MLP_PRED_PATH, "mlp_pred"),
        (GRU_PRED_PATH, "gru_mlp_pred"),
        (ONLINE_MLP_PRED_PATH, "online_mlp_pred"),
        (ONLINE_GRU_PRED_PATH, "online_gru_mlp_pred"),
    ]:
        next_df = load_nn_predictions(path, pred_col)
        if next_df is not None:
            pred_df = pred_df.merge(next_df, on=keys, how="inner")
            pred_cols.append(pred_col)

    if len(pred_cols) < 2:
        raise ValueError("Need at least xgb plus one neural prediction file for ensemble.")

    single_scores = score_frame(pred_df, pred_cols)
    valid_df = pred_df[pred_df["split"] == "valid"].copy()
    weights_df, best = tune_blend(valid_df, pred_cols)
    weights_df.to_csv(WEIGHTS_PATH, index=False)

    for split in ["valid", "test"]:
        mask = pred_df["split"] == split
        pred_df.loc[mask, "ensemble_pred"] = sum(
            best[f"w_{col}"] * pred_df.loc[mask, col] for col in pred_cols
        )

    ensemble_scores = score_frame(pred_df, ["ensemble_pred"])
    all_scores = pd.concat([single_scores, ensemble_scores], ignore_index=True)
    pred_df.to_parquet(PRED_PATH, index=False)

    report = {
        "prediction_path": str(PRED_PATH),
        "weights_path": str(WEIGHTS_PATH),
        "best_weights": best,
        "scores": all_scores.to_dict(orient="records"),
        "rows": {split: int((pred_df["split"] == split).sum()) for split in ["valid", "test"]},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print("DONE", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
