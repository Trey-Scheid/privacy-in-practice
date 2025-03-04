import duckdb
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def load_and_query_parquet(parquet_path_pattern, epsilon_values):
    parquet_files = glob.glob(parquet_path_pattern)
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {parquet_path_pattern}")
    
    print(f"Found {len(parquet_files)} Parquet files. Processing...")
    for file in parquet_files:
        print(f"Processing {os.path.basename(file)}...")

    # Convert list to a format that DuckDB understands
    parquet_file_list = ", ".join(f"'{file}'" for file in parquet_files)

    guid_count_query = f"SELECT COUNT(DISTINCT guid) FROM read_parquet({parquet_files});"
    num_guids = duckdb.query(guid_count_query).fetchone()[0]  # Fetch the total count

    # Step 2: Define epsilon values
    epsilon_values = epsilon_values 

    # Step 3: Compute sensitivity dynamically
    sensitivity_percentage = 1    

    # Store loss values
    loss_values = []

    for i, epsilon in enumerate(epsilon_values):
        # Compute Laplace scale
        b = sensitivity_percentage / epsilon

        # DuckDB SQL Query to count event occurrences
        query = f"""
        WITH guid_counts AS (
            SELECT 
                guid,
                COUNT(CASE WHEN event_id = '19' THEN 1 END) AS e19_count,
                SUM(CASE WHEN event_id LIKE '%1001%' THEN 1 ELSE 0 END) AS num_ones_1001,
                SUM(CASE WHEN event_id LIKE '%41%' THEN 1 ELSE 0 END) AS num_ones_41
            FROM read_parquet({parquet_files})
            GROUP BY guid
        ),
        histogram AS (
            SELECT 
                CASE 
                    WHEN e19_count >= 30 THEN '30+'
                    ELSE CAST(e19_count AS VARCHAR)
                END AS bin,
                SUM(num_ones_1001) AS num_ones_1001,
                COUNT(*) - SUM(num_ones_1001) AS num_zeros_1001,
                SUM(num_ones_41) AS num_ones_41,
                COUNT(*) - SUM(num_ones_41) AS num_zeros_41
            FROM guid_counts
            GROUP BY bin
        )
        SELECT 
            bin,
            num_ones_1001,
            num_zeros_1001,
            num_ones_41,
            num_zeros_41
        FROM histogram
        ORDER BY 
            CASE 
                WHEN bin = '30+' THEN 30 
                ELSE CAST(bin AS INTEGER) 
            END;
        """

        con = duckdb.connect()
        df_original = con.execute(query).fetch_df()
        con.close()

        # Add Laplace noise to both counts
        df_noisy = df_original.copy()
        df_noisy['num_ones_1001'] += np.random.laplace(0, b, len(df_noisy))
        df_noisy['num_zeros_1001'] += np.random.laplace(0, b, len(df_noisy))
        df_noisy['num_ones_41'] += np.random.laplace(0, b, len(df_noisy))
        df_noisy['num_zeros_41'] += np.random.laplace(0, b, len(df_noisy))

        # Compute percentages
        df_original['percent_with_1001'] = (df_original['num_ones_1001'] / (df_original['num_ones_1001'] + df_original['num_zeros_1001'])) * 100
        df_original['percent_with_41'] = (df_original['num_ones_41'] / (df_original['num_ones_41'] + df_original['num_zeros_41'])) * 100

        df_noisy['percent_with_1001'] = (df_noisy['num_ones_1001'] / (df_noisy['num_ones_1001'] + df_noisy['num_zeros_1001'])) * 100
        df_noisy['percent_with_41'] = (df_noisy['num_ones_41'] / (df_noisy['num_ones_41'] + df_noisy['num_zeros_41'])) * 100

        # Compute Mean Absolute Error (MAE)
        loss_1001 = np.mean(np.abs(df_original['percent_with_1001'] - df_noisy['percent_with_1001']))
        loss_41 = np.mean(np.abs(df_original['percent_with_41'] - df_noisy['percent_with_41']))
        
        loss_values.append((loss_1001 + loss_41) / 2)  # Average loss
