import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import duckdb


def load_and_query_parquet(parquet_path_pattern):
    """Loads multiple Parquet files and executes a DuckDB query to process them."""
    parquet_files = glob.glob(parquet_path_pattern)
    
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {parquet_path_pattern}")
    
    print(f"Found {len(parquet_files)} Parquet files. Processing...")

    # Print progress for each file
    for file in parquet_files:
        print(f"Processing {os.path.basename(file)}...")

    # Convert list to a format that DuckDB understands
    parquet_file_list = ", ".join(f"'{file}'" for file in parquet_files)

    query = f"""
    WITH guid_counts AS (
        SELECT 
            guid,
            COUNT(CASE WHEN event_id = '19' THEN 1 END) AS e19_count,
            MAX(CASE WHEN event_id LIKE '%1001%' THEN 1 ELSE 0 END) AS has_1001,
            MAX(CASE WHEN event_id LIKE '%41%' THEN 1 ELSE 0 END) AS has_41
        FROM read_parquet([{parquet_file_list}])
        GROUP BY guid
    ),
    histogram AS (
        SELECT 
            CASE 
                WHEN e19_count >= 30 THEN '30+'
                ELSE CAST(e19_count AS VARCHAR)
            END AS bin,
            AVG(has_1001) * 100 AS percent_with_1001,
            AVG(has_41) * 100 AS percent_with_41
        FROM guid_counts
        GROUP BY bin
    )
    SELECT 
        bin,
        percent_with_1001,
        percent_with_41
    FROM histogram
    ORDER BY 
        CASE 
            WHEN bin = '30+' THEN 30 
            ELSE CAST(bin AS INTEGER) 
        END;
    """

    return duckdb.query(query).to_df()

def plot_histogram(df):
    """Plots the histogram from the query results."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['bin'], df['percent_with_1001'], marker='o', color='orange', label='Percentage with 1001')
    plt.plot(df['bin'], df['percent_with_41'], marker='o', color='green', label='Percentage with 41')

    plt.xlabel('E19 Count Bins')
    plt.ylabel('Percentage of GUIDs (%)')
    plt.title('Percentage of GUIDs with Event ID 1001 and 41 by E19 Occurrences')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig("nonprivatehist.pdf")
    plt.show()
    

def main():
    """Main execution function."""
    parquet_path_pattern = "D:/CapstoneTyler/*.parquet"
    
    try:
        df = load_and_query_parquet(parquet_path_pattern)
        plot_histogram(df)
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()