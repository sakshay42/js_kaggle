from pathlib import Path
import json

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "clean" / "partition_8_drop_missing" / "part-0.parquet"
TOP20_PATH = PROJECT_ROOT / "models" / "lgbm_baseline_top20" / "top20_features.txt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "top20_p8"
REPORT_PATH = PROJECT_ROOT / "logs" / "make_top20_p8_report.json"

DATE_COL = "date_id"
TIME_COL = "time_id"
SYMBOL_COL = "symbol_id"
WEIGHT_COL = "weight"
TARGET_COL = "responder_6"
BASE_COLS = [DATE_COL, TIME_COL, SYMBOL_COL, WEIGHT_COL, TARGET_COL]

VALID_DATE_COUNT = 20
TEST_DATE_COUNT = 20
BATCH_SIZE = 65_536


def unique_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def load_top20_features():
    if not TOP20_PATH.exists():
        raise FileNotFoundError(f"Missing top-20 feature file: {TOP20_PATH}")

    features = [line.strip() for line in TOP20_PATH.read_text().splitlines() if line.strip()]
    if len(features) != 20:
        raise ValueError(f"Expected 20 features, found {len(features)} in {TOP20_PATH}")
    return features


def collect_dates(dataset):
    dates = set()
    scanner = dataset.scanner(
        columns=[DATE_COL],
        batch_size=BATCH_SIZE,
        batch_readahead=1,
        fragment_readahead=1,
        use_threads=True,
    )
    for batch in scanner.to_batches():
        values = batch.column(0).drop_null().unique().to_pylist()
        dates.update(values)
    return sorted(dates)


def make_writer(path, schema):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return pq.ParquetWriter(path, schema=schema, compression="zstd", use_dictionary=True)


def write_if_rows(writer, table):
    if table.num_rows:
        writer.write_table(table)
    return table.num_rows


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing cleaned parquet file: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = ds.dataset(INPUT_PATH, format="parquet")
    schema_names = dataset.schema.names
    top20_features = load_top20_features()
    output_cols = unique_preserve_order(BASE_COLS + top20_features)

    missing_cols = [col for col in output_cols if col not in schema_names]
    if missing_cols:
        raise ValueError(f"Missing required columns in cleaned data: {missing_cols}")

    dates = collect_dates(dataset)
    if len(dates) <= VALID_DATE_COUNT + TEST_DATE_COUNT:
        raise ValueError("Not enough dates for train/valid/test split")

    test_start = dates[-TEST_DATE_COUNT]
    valid_start = dates[-(TEST_DATE_COUNT + VALID_DATE_COUNT)]

    output_schema = pa.schema([dataset.schema.field(col) for col in output_cols])
    paths = {
        "train": OUTPUT_DIR / "train.parquet",
        "valid": OUTPUT_DIR / "valid.parquet",
        "test": OUTPUT_DIR / "test.parquet",
    }
    writers = {name: make_writer(path, output_schema) for name, path in paths.items()}
    rows = {"train": 0, "valid": 0, "test": 0}

    scanner = dataset.scanner(
        columns=output_cols,
        batch_size=BATCH_SIZE,
        batch_readahead=1,
        fragment_readahead=1,
        use_threads=True,
    )

    try:
        for batch_idx, batch in enumerate(scanner.to_batches(), start=1):
            table = pa.Table.from_batches([batch], schema=output_schema)
            date_values = table[DATE_COL]

            train_mask = pc.less(date_values, valid_start)
            valid_mask = pc.and_(
                pc.greater_equal(date_values, valid_start),
                pc.less(date_values, test_start),
            )
            test_mask = pc.greater_equal(date_values, test_start)

            rows["train"] += write_if_rows(writers["train"], table.filter(train_mask))
            rows["valid"] += write_if_rows(writers["valid"], table.filter(valid_mask))
            rows["test"] += write_if_rows(writers["test"], table.filter(test_mask))

            if batch_idx % 25 == 0:
                print(
                    f"batch {batch_idx:,} | "
                    f"train {rows['train']:,} | valid {rows['valid']:,} | test {rows['test']:,}",
                    flush=True,
                )
    finally:
        for writer in writers.values():
            writer.close()

    report = {
        "input_path": str(INPUT_PATH),
        "top20_path": str(TOP20_PATH),
        "output_dir": str(OUTPUT_DIR),
        "columns": output_cols,
        "top20_features": top20_features,
        "date_split": {
            "date_min": int(dates[0]),
            "date_max": int(dates[-1]),
            "unique_dates": int(len(dates)),
            "train_date_min": int(dates[0]),
            "train_date_max": int(valid_start - 1),
            "valid_date_min": int(valid_start),
            "valid_date_max": int(test_start - 1),
            "test_date_min": int(test_start),
            "test_date_max": int(dates[-1]),
            "valid_date_count": VALID_DATE_COUNT,
            "test_date_count": TEST_DATE_COUNT,
        },
        "rows": rows,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("DONE", flush=True)
    print(f"report={REPORT_PATH}", flush=True)
    print(f"output_dir={OUTPUT_DIR}", flush=True)
    print(f"columns={len(output_cols)}", flush=True)
    print(f"train_rows={rows['train']:,}", flush=True)
    print(f"valid_rows={rows['valid']:,}", flush=True)
    print(f"test_rows={rows['test']:,}", flush=True)


if __name__ == "__main__":
    main()
