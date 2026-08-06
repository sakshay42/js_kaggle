from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "top20_rolling_p8"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = MODEL_DIR / "ensemble" / "focused_mlp_xgb"
LOG_DIR = PROJECT_ROOT / "logs"

XGB_MODEL_PATH = MODEL_DIR / "xgboost" / "top20_rolling_xgb" / "best_xgb.json"
FOCUSED_PRED_PATH = MODEL_DIR / "neural" / "regularized_mlp_focused" / "predictions.parquet"
REPORT_PATH = LOG_DIR / "ensemble_focused_mlp_xgb_report.json"
WEIGHTS_PATH = OUT_DIR / "blend_weights.csv"
PRED_PATH = OUT_DIR / "predictions.parquet"

TARGET_COL = "responder_6"
WEIGHT_COL = "weight"
DATE_COL = "date_id"
TOP_N_FOCUSED_MODELS = 8


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
    df = pd.read_parquet(DATA_DIR / f"{split}.parquet", engine="pyarrow")
    features = feature_cols(df)
    model = xgb.Booster()
    model.load_model(str(XGB_MODEL_PATH))
    dm = xgb.DMatrix(xgb_feature_frame(df, features), feature_names=features)
    return df[[DATE_COL, "time_id", "symbol_id", TARGET_COL, WEIGHT_COL]].assign(
        split=split,
        xgb_pred=model.predict(dm),
    )


def load_focused_predictions():
    if not FOCUSED_PRED_PATH.exists():
        raise FileNotFoundError(FOCUSED_PRED_PATH)
    pred = pd.read_parquet(FOCUSED_PRED_PATH, engine="pyarrow")

    valid = pred[pred["split"] == "valid"]
    scores = []
    for model_name, group in valid.groupby("model"):
        scores.append(
            {
                "model": model_name,
                "valid_weighted_r2": weighted_r2(group[TARGET_COL], group["prediction"], group[WEIGHT_COL]),
            }
        )
    score_df = pd.DataFrame(scores).sort_values("valid_weighted_r2", ascending=False)
    keep_models = score_df.head(TOP_N_FOCUSED_MODELS)["model"].tolist()
    pred = pred[pred["model"].isin(keep_models)].copy()

    keys = [DATE_COL, "time_id", "symbol_id", "split"]
    wide = pred.pivot_table(index=keys, columns="model", values="prediction", aggfunc="first").reset_index()
    wide.columns.name = None
    wide["focused_mlp_mean_pred"] = wide[keep_models].mean(axis=1)
    return wide[keys + ["focused_mlp_mean_pred"]], score_df


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


def tune_two_model_blend(valid_df):
    rows = []
    for w_xgb in np.arange(0.0, 1.01, 0.025):
        w_mlp = 1.0 - w_xgb
        pred = w_xgb * valid_df["xgb_pred"] + w_mlp * valid_df["focused_mlp_mean_pred"]
        rows.append(
            {
                "w_xgb_pred": float(w_xgb),
                "w_focused_mlp_mean_pred": float(w_mlp),
                "weighted_r2": weighted_r2(valid_df[TARGET_COL], pred, valid_df[WEIGHT_COL]),
            }
        )
    weights = pd.DataFrame(rows).sort_values("weighted_r2", ascending=False)
    return weights, weights.iloc[0].to_dict()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.concat([add_xgb_preds("valid"), add_xgb_preds("test")], ignore_index=True)
    focused, focused_scores = load_focused_predictions()
    keys = [DATE_COL, "time_id", "symbol_id", "split"]
    pred_df = base.merge(focused, on=keys, how="inner")

    weights_df, best = tune_two_model_blend(pred_df[pred_df["split"] == "valid"])
    weights_df.to_csv(WEIGHTS_PATH, index=False)

    pred_df["ensemble_pred"] = (
        best["w_xgb_pred"] * pred_df["xgb_pred"]
        + best["w_focused_mlp_mean_pred"] * pred_df["focused_mlp_mean_pred"]
    )

    scores = score_frame(pred_df, ["xgb_pred", "focused_mlp_mean_pred", "ensemble_pred"])
    pred_df.to_parquet(PRED_PATH, index=False)

    report = {
        "prediction_path": str(PRED_PATH),
        "weights_path": str(WEIGHTS_PATH),
        "focused_prediction_path": str(FOCUSED_PRED_PATH),
        "top_n_focused_models": TOP_N_FOCUSED_MODELS,
        "best_weights": best,
        "scores": scores.to_dict(orient="records"),
        "focused_model_scores": focused_scores.head(TOP_N_FOCUSED_MODELS).to_dict(orient="records"),
        "rows": {split: int((pred_df["split"] == split).sum()) for split in ["valid", "test"]},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print("DONE", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
