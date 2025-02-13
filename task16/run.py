from src.kmeans import DeviceUsage

def run_device_usage_analysis(db_path, parquet_file):
    """Run the device usage analysis pipeline.

    Args:
        db_path (str): Path to the DuckDB database file.
        parquet_file (str): Path to the input Parquet file.

    Returns:
        pd.DataFrame: DataFrame containing the clustered device data.
    """
    try:
        # Initialize the analyzer object with the provided database and parquet file
        analyzer = DeviceUsage(db_path=db_path, parquet_file=parquet_file)

        # Run the full analysis pipeline
        result = analyzer.run_analysis()

        # Return the analysis result
        return result
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

def main():
    """Main function to execute the device usage analysis script."""
    db_path = 'duckdb.db'  # Path to DuckDB database
    parquet_file = 'Novel-Techniques-in-Private-Data-Analysis/task16/data/raw/0007_part_09_limit_1000000.parquet'  # Path to Parquet data file

    print("Starting device usage analysis...")

    # Run the analysis
    result = run_device_usage_analysis(db_path, parquet_file)

    # Show the result
    print("Analysis complete. Here is the result:")
    print(result.head())

if __name__ == "__main__":
    # Execute the main function
    main()
