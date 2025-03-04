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