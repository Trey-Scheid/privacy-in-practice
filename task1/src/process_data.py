import pyarrow.compute
import pyarrow.dataset as ds
import pyarrow as pa
import duckdb
import glob
import os
import json
from dotenv import load_dotenv
from utils import sample_table

load_dotenv()

ITEM_DIR = os.getenv("ITEM_DIR")
HEADER_DIR = os.getenv("HEADER_DIR")
CHECKPOINT_FILE = os.getenv("CHECKPOINT_FILE")
DUCK_TEMP_DIR = os.getenv("DUCK_TEMP_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")


def raw_to_aggregated(con, item_dir, header_dir, output_dir, checkpoint_file):
    """
    Convert raw data from Intel Telemetry dataset to aggregated data with correct schema.

    Args:
        con (duckdb.Connection): DuckDB connection.
        item_dir (str): Path to the item directory.
        header_dir (str): Path to the header directory.
        output_dir (str): Path to the output directory.
        checkpoint_file (str): Path to the checkpoint file.
    """

    processed_combinations = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            processed_combinations = set(tuple(x) for x in json.load(f))
    else:
        with open(checkpoint_file, "w") as f:
            json.dump([], f)

    header_files = glob.glob(os.path.join(header_dir, "*.parquet"))
    item_files = glob.glob(os.path.join(item_dir, "*.parquet"))

    count_to_process = len(header_files) * len(item_files)
    files_processed = len(processed_combinations)

    if files_processed >= count_to_process:
        print(f"All {count_to_process} files have been processed.")
        return

    for item_file in item_files:
        for header_file in header_files:
            if (header_file, item_file) in processed_combinations:
                continue

            print(
                f"Processing {files_processed} of {count_to_process}: {header_file} + {item_file}"
            )

            files_processed += 1
            # First check how many rows we'll get
            try:
                count_query = f"""
                    SELECT COUNT(*)
                    FROM (
                        SELECT header.create_dt,
                        FROM read_parquet('{header_file}') AS header
                        JOIN read_parquet('{item_file}') AS item
                        ON header.hash = item.hash
                        WHERE 
                            header.create_dt >= '2020-02-01'
                            AND header.create_dt < '2021-02-01'
                        GROUP BY header.create_dt, header.guid
                        LIMIT 1
                    )
                """
                row_exists = con.execute(count_query).fetchone()[0] > 0

                processed_combinations.add((header_file, item_file))
                with open(checkpoint_file, "w") as f:
                    json.dump([list(x) for x in processed_combinations], f)

                if not row_exists:
                    continue

                output_file = f"{output_dir}/final_dataset_{os.path.basename(header_file)}_{os.path.basename(item_file)}.parquet"

                # If we have rows, copy them to a file
                copy_query = f"""
                    COPY (
                        SELECT
                            header.create_dt,
                            header.guid,
                            MAX(CASE WHEN header.event_id = 19 THEN 1 ELSE 0 END) as has_corrected_error,
                            MAX(CASE WHEN header.event_id = 41 THEN 1 ELSE 0 END) as has_bugcheck,
                            MAX(CASE 
                                WHEN header.event_id = 41 AND item.key_name = 'BugcheckCode' 
                                THEN item.value 
                                ELSE NULL 
                            END) as bugcheck_code
                        FROM read_parquet('{header_file}') AS header
                        JOIN read_parquet('{item_file}') AS item
                        ON header.hash = item.hash
                        WHERE 
                            header.create_dt >= '2020-02-01'
                            AND header.create_dt < '2021-02-01'
                        GROUP BY header.create_dt, header.guid
                    ) TO '{output_file}' (FORMAT 'parquet')
                """
                con.execute(copy_query)
                print(f"Saved results to {output_file}")

            except Exception as e:
                print(f"Error processing {header_file} + {item_file}: {e}")
                break


def aggregated_to_final(con, output_dir, data_dir="data/"):
    """
    Convert aggregated data to final data. Top 30 most common bugcheck codes and downsampled to 5:1 ratio of no bugcheck:bugcheck

    Args:
        con (duckdb.Connection): DuckDB connection.
        output_dir (str): Path to the output directory.
        data_dir (str): Path to the data directory.
    """
    query = f"""
        SELECT bugcheck_code
        FROM '{output_dir}/*.parquet'
        GROUP BY bugcheck_code
        ORDER BY count(*) DESC
        LIMIT 31
    """

    top_30 = con.execute(query).fetchall()
    top_30 = [x[0] for x in top_30]

    dataset = ds.dataset(output_dir, format="parquet")
    table = dataset.to_table(
        columns=["has_corrected_error", "bugcheck_code", "has_bugcheck"]
    )

    for bugcheck_code in top_30:
        if bugcheck_code is None:
            continue
        print("processing ", bugcheck_code)

        filtered_table_true = table.filter(
            pyarrow.compute.equal(table["bugcheck_code"], bugcheck_code)
        )
        num_rows_true = filtered_table_true.num_rows
        filtered_table_false = sample_table(
            table.filter(pyarrow.compute.equal(table["has_bugcheck"], 0)),
            num_rows_true * 5,
        )
        filtered_table = pa.concat_tables([filtered_table_true, filtered_table_false])

        df_filtered = filtered_table.to_pandas()

        output_file = os.path.join(data_dir, f"bugcheck_{bugcheck_code}.csv")
        df_filtered.to_csv(output_file, index=False)


def main():
    con = duckdb.connect()
    con.execute(f"PRAGMA temp_directory='{DUCK_TEMP_DIR}';")

    raw_to_aggregated(con, ITEM_DIR, HEADER_DIR, OUTPUT_DIR, CHECKPOINT_FILE)
    aggregated_to_final(con, OUTPUT_DIR)


if __name__ == "__main__":
    main()
