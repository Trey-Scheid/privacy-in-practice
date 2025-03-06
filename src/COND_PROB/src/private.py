import duckdb
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd

def caluculate_utility(parquet_path_pattern, epsilon_values):
    parquet_files = glob.glob(parquet_path_pattern)
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {parquet_path_pattern}")
    
    print(f"Found {len(parquet_files)} Parquet files. Processing...")
    for file in parquet_files:
        print(f"Processing {os.path.basename(file)}...")


    parquet_file_list = ", ".join(f"'{file}'" for file in parquet_files)

    guid_count_query = f"SELECT COUNT(DISTINCT guid) FROM read_parquet({parquet_files});"
    num_guids = duckdb.query(guid_count_query).fetchone()[0]  # Fetch the total count

    # Step 2: Define epsilon values and ensure ε = 1 is included
    epsilon_values = epsilon_values

    # Step 3: Compute sensitivity dynamically
    sensitivity_percentage = 1   

    # Store loss values
    loss_values = []

    for epsilon in epsilon_values:
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

    min_loss = min(loss_values)
    max_loss = max(loss_values)
    normalized_loss = [(loss - min_loss) / (max_loss - min_loss) for loss in loss_values]

    # Compute utility as the inverse of normalized loss

    utility_values = [1 - nl for nl in normalized_loss]  
    return [pd.DataFrame({'task': ["COND_PROB" for i in range(len(epsilon_values))],
                         'epsilon': epsilon_values,
                         'utility': utility_values})]


def plot_histogram(parquet_path_pattern, epsilon, output_dir):
    parquet_files = glob.glob(parquet_path_pattern)
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {parquet_path_pattern}")
    
    print(f"Found {len(parquet_files)} Parquet files. Processing...")
    for file in parquet_files:
        print(f"Processing {os.path.basename(file)}...")


    parquet_file_list = ", ".join(f"'{file}'" for file in parquet_files)

    guid_count_query = f"SELECT COUNT(DISTINCT guid) FROM read_parquet({parquet_files});"
    num_guids = duckdb.query(guid_count_query).fetchone()[0]  # Fetch the total count

    sensitivity_percentage = 1   

    # Store loss values
    loss_values = []
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


    plt.figure(figsize=(8, 6))
    plt.plot(df_original['bin'], df_original['percent_with_1001'], alpha=0.6, label='Original 1001', color='blue')
    plt.plot(df_original['bin'], df_original['percent_with_41'], alpha=0.6, label='Original 41', color='green')

    plt.plot(df_noisy['bin'], df_noisy['percent_with_1001'], alpha=0.6, label='Noisy 1001', color='red', linestyle='dashed')
    plt.plot(df_noisy['bin'], df_noisy['percent_with_41'], alpha=0.6, label='Noisy 41', color='orange', linestyle='dashed')

    plt.xlabel("Event Count Bins", color='black', fontsize=12)
    plt.ylabel("Percentage", color='black', fontsize=12)
    plt.title(f"Epsilon = {epsilon} | Noisy vs. Original Histograms", fontsize=14, color='black')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # Save figure
    plot_filename = os.path.join(output_dir, "epsilon_vs_utility.png")
    plt.savefig(plot_filename, dpi=300, facecolor='white')
    plt.close()

def combine_parquets(parquetlist):
    parquet_file_list = ", ".join(f"'{file}'" for file in parquetlist)



def main():
    """Main execution function."""


    #This does 1 parquet
    parquet_path_pattern = "D:/CapstoneTyler/0007_part_00.parquet"


    #This does all parquets in my hard drive
    #parquet_path_pattern = "D:/CapstoneTyler/*.parquet"

    #This does the dummy parquets
    #parquet_path_pattern = r"C:\Users\Tyler\OneDrive\Desktop\180B\Novel-Techniques-in-Private-Data-Analysis\dummy_data\eventlog_header_hist_CONDPROB\header.parquet"

    #This does a select few parquets
    #parquet_path_pattern =  [r"D:\CapstoneTyler\0007_part_00.parquet",r"D:\CapstoneTyler\0007_part_36.parquet",r"D:\CapstoneTyler\0007_part_38.parquet",r"D:\CapstoneTyler\0007_part_39.parquet",r"D:\CapstoneTyler\0007_part_40.parquet",
    #r"D:\CapstoneTyler\0007_part_42.parquet",r"D:\CapstoneTyler\0003_part_00.parquet", r"D:\CapstoneTyler\0003_part_36.parquet",r"D:\CapstoneTyler\0003_part_37.parquet"]

    

    try:

        #This calcualtes the utilty
        print(caluculate_utility(parquet_path_pattern, [0.01,.1,1,10,100]))

        #This plots the histogram at a certain epsilon
        #plot = plot_histogram(parquet_path_pattern, 1, output_dir=r"D:\CapstoneTyler\visualizations")

    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
