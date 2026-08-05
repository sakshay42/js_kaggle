from pathlib import Path
import json
import gc

import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "clean" / "partition_8_drop_missing"
REPORT_PATH = PROJECT_ROOT / "logs" / "validate_p8_report.json"

TARGET_COL = "responder_6"
WEIGHT_COL = "weight"
ID_COLS = ["date_id", "time_id", "symbol_id"]
DISCRETE_COLS = ["symbol_id", "time_id", "feature_09", "feature_10", "feature_11"]
BATCH_SIZE = 8192


def json_scalar(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def empty_range():
    return {"min": None, "max": None, "unique_count": None}


def update_min_max(current, values):
    if len(values) == 0:
        return current
    v_min = values.min()
    v_max = values.max()
    v_min = json_scalar(v_min)
    v_max = json_scalar(v_max)
    current["min"] = v_min if current["min"] is None else min(current["min"], v_min)
    current["max"] = v_max if current["max"] is None else max(current["max"], v_max)
    return current


def main():
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Missing cleaned parquet dataset: {CLEAN_PATH}")

    dataset = ds.dataset(CLEAN_PATH, format="parquet")
    columns = dataset.schema.names
    feature_cols = [col for col in columns if col.startswith("feature_")]
    present_id_cols = [col for col in ID_COLS if col in columns]
    present_discrete_cols = [col for col in DISCRETE_COLS if col in columns]
    required_cols = feature_cols + [TARGET_COL, WEIGHT_COL]

    missing_required = [col for col in [TARGET_COL, WEIGHT_COL] if col not in columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    if not feature_cols:
        raise ValueError("No feature_ columns found")

    scan_cols = present_id_cols + feature_cols + [WEIGHT_COL, TARGET_COL]
    scanner = dataset.scanner(
        columns=scan_cols,
        batch_size=BATCH_SIZE,
        batch_readahead=1,
        fragment_readahead=1,
        use_threads=False,
    )

    rows = 0
    missing_required_values = 0
    id_ranges = {col: empty_range() for col in present_id_cols}
    id_unique_values = {col: set() for col in present_id_cols}
    discrete_unique_values = {col: set() for col in present_discrete_cols}

    for batch_idx, record_batch in enumerate(scanner.to_batches(), start=1):
        df = record_batch.to_pandas(split_blocks=True, self_destruct=True)
        rows += len(df)
        missing_required_values += int(df[required_cols].isna().sum().sum())

        for col in present_id_cols:
            non_null = df[col].dropna()
            id_ranges[col] = update_min_max(id_ranges[col], non_null)
            id_unique_values[col].update(non_null.unique().tolist())

        for col in present_discrete_cols:
            discrete_unique_values[col].update(df[col].dropna().unique().tolist())

        if batch_idx % 100 == 0:
            print(f"validated batches: {batch_idx:,} | rows: {rows:,}", flush=True)

        del df
        gc.collect()

    for col in present_id_cols:
        id_ranges[col]["unique_count"] = len(id_unique_values[col])

    discrete_summary = {
        col: {
            "unique_count": len(values),
            "values": [json_scalar(value) for value in sorted(values)],
        }
        for col, values in discrete_unique_values.items()
    }

    schema_summary = {
        field.name: str(field.type)
        for field in dataset.schema
    }

    report = {
        "clean_path": str(CLEAN_PATH),
        "rows": rows,
        "columns": len(columns),
        "feature_count": len(feature_cols),
        "missing_values_in_features_target_weight": missing_required_values,
        "id_ranges": id_ranges,
        "discrete_columns": present_discrete_cols,
        "discrete_summary": discrete_summary,
        "schema": schema_summary,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("DONE")
    print(f"report={REPORT_PATH}")
    print(f"rows={rows:,}")
    print(f"columns={len(columns):,}")
    print(f"feature_count={len(feature_cols):,}")
    print(f"missing_values_in_features_target_weight={missing_required_values:,}")
    print("id_ranges:")
    for col, values in id_ranges.items():
        print(f"  {col}: {values}")
    print("discrete_summary:")
    for col, values in discrete_summary.items():
        print(f"  {col}: {values}")


if __name__ == "__main__":
    main()
