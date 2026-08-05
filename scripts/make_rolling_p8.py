from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "top20_p8"
OUTPUT_DIR = PROJECT_ROOT / "data" / "top20_rolling_p8"
REPORT_PATH = PROJECT_ROOT / "logs" / "make_rolling_p8_report.json"

DATE_COL = "date_id"
TIME_COL = "time_id"
SYMBOL_COL = "symbol_id"
TARGET_COL = "responder_6"
WEIGHT_COL = "weight"
SPLIT_COL = "_split"

ROLL_WINDOWS = [5, 20]
ROLL_BASE_FEATURES = [
    "feature_08",
    "feature_36",
    "feature_61",
    "feature_04",
    "feature_20",
    "feature_06",
    "feature_24",
    "feature_29",
    "feature_58",
    "feature_23",
]


def load_split(name):
    path = INPUT_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing input split: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    df[SPLIT_COL] = name
    print(f"{name}_shape={df.shape}", flush=True)
    return df


def add_symbol_rolling_features(df, features, windows):
    df = df.sort_values([SYMBOL_COL, DATE_COL, TIME_COL]).reset_index(drop=True)
    created = []
    grouped = df.groupby(SYMBOL_COL, observed=True, sort=False)

    for feature in features:
        shifted = grouped[feature].shift(1)
        for window in windows:
            mean_col = f"{feature}_sym_roll{window}_mean"
            std_col = f"{feature}_sym_roll{window}_std"
            rolled = shifted.groupby(df[SYMBOL_COL], observed=True)
            df[mean_col] = (
                rolled.rolling(window, min_periods=2).mean().reset_index(level=0, drop=True)
            )
            df[std_col] = (
                rolled.rolling(window, min_periods=2).std().reset_index(level=0, drop=True)
            )
            created.extend([mean_col, std_col])

    df[created] = df[created].fillna(0).astype("float32")
    df = df.sort_values([SPLIT_COL, DATE_COL, TIME_COL, SYMBOL_COL]).reset_index(drop=True)
    return df, created


def write_split(df, split):
    out = df[df[SPLIT_COL] == split].drop(columns=[SPLIT_COL]).copy()
    path = OUTPUT_DIR / f"{split}.parquet"
    out.to_parquet(path, index=False, compression="zstd")
    return path, len(out)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.concat([load_split("train"), load_split("valid"), load_split("test")], ignore_index=True)
    roll_features = [col for col in ROLL_BASE_FEATURES if col in df.columns]
    missing = [col for col in ROLL_BASE_FEATURES if col not in df.columns]
    if missing:
        print(f"missing_roll_features={missing}", flush=True)

    print(f"combined_shape={df.shape}", flush=True)
    df, created = add_symbol_rolling_features(df, roll_features, ROLL_WINDOWS)
    print(f"rolling_feature_count={len(created)}", flush=True)

    outputs = {}
    rows = {}
    for split in ["train", "valid", "test"]:
        path, row_count = write_split(df, split)
        outputs[split] = str(path)
        rows[split] = int(row_count)
        print(f"{split}_output={path} rows={row_count:,}", flush=True)

    report = {
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "roll_windows": ROLL_WINDOWS,
        "roll_base_features": roll_features,
        "rolling_features": created,
        "rows": rows,
        "outputs": outputs,
        "columns": [col for col in df.columns if col != SPLIT_COL],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("DONE", flush=True)
    print(f"report={REPORT_PATH}", flush=True)
    print(f"output_dir={OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
