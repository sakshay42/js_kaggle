from pathlib import Path
import gc
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PARTITION8_PATH = DATA_DIR / "part_8.parquet"
OUTPUT_DIR = DATA_DIR / "clean" / "partition_8_drop_missing"

TARGET_COL = "responder_6"
WEIGHT_COL = "weight"
ID_COLS = ["date_id", "time_id", "symbol_id"]
DISCRETE_COLS = ["symbol_id", "time_id", "feature_09", "feature_10", "feature_11"]

BATCH_SIZE = 8192
OVERWRITE_OUTPUT = True


def cast_clean_batch(batch_df, feature_cols, discrete_cols):
    for col in feature_cols:
        if col in discrete_cols:
            continue
        batch_df[col] = batch_df[col].astype(np.float32)

    batch_df[WEIGHT_COL] = batch_df[WEIGHT_COL].astype(np.float32)
    batch_df[TARGET_COL] = batch_df[TARGET_COL].astype(np.float32)

    for col in discrete_cols:
        if col in batch_df.columns:
            batch_df[col] = batch_df[col].astype("int16")

    if "date_id" in batch_df.columns:
        batch_df["date_id"] = batch_df["date_id"].astype("int16")

    return batch_df


def main():
    if not PARTITION8_PATH.exists():
        raise FileNotFoundError(f"Missing input parquet: {PARTITION8_PATH}")

    dataset = ds.dataset(PARTITION8_PATH, format="parquet")
    columns = dataset.schema.names

    feature_cols = [col for col in columns if col.startswith("feature_")]
    present_id_cols = [col for col in ID_COLS if col in columns]
    present_discrete_cols = [col for col in DISCRETE_COLS if col in columns]
    selected_cols = present_id_cols + [WEIGHT_COL] + feature_cols + [TARGET_COL]
    drop_missing_cols = feature_cols + [TARGET_COL, WEIGHT_COL]

    if OUTPUT_DIR.exists():
        if OVERWRITE_OUTPUT:
            shutil.rmtree(OUTPUT_DIR)
        else:
            raise FileExistsError(f"Output already exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "part-0.parquet"
    scanner = dataset.scanner(
        columns=selected_cols,
        batch_size=BATCH_SIZE,
        batch_readahead=1,
        fragment_readahead=1,
        use_threads=False,
    )

    writer = None
    rows_read = 0
    rows_written = 0
    rows_dropped = 0

    try:
        for batch_idx, record_batch in enumerate(scanner.to_batches(), start=1):
            batch_df = record_batch.to_pandas(split_blocks=True, self_destruct=True)
            rows_read += len(batch_df)

            clean_df = batch_df.dropna(subset=drop_missing_cols).copy()
            rows_dropped += len(batch_df) - len(clean_df)

            if clean_df.empty:
                del batch_df, clean_df
                gc.collect()
                continue

            clean_df = cast_clean_batch(clean_df, feature_cols, present_discrete_cols)
            table = pa.Table.from_pandas(clean_df[selected_cols], preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema, compression="zstd")

            writer.write_table(table)
            rows_written += len(clean_df)

            if batch_idx % 25 == 0:
                print(
                    f"batch {batch_idx:,} | read {rows_read:,} | "
                    f"written {rows_written:,} | dropped {rows_dropped:,}",
                    flush=True,
                )

            del batch_df, clean_df, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    print("DONE")
    print(f"output_file={output_file}")
    print(f"rows_read={rows_read:,}")
    print(f"rows_written={rows_written:,}")
    print(f"rows_dropped={rows_dropped:,}")
    print(f"drop_rate={rows_dropped / rows_read:.6f}")


if __name__ == "__main__":
    main()
