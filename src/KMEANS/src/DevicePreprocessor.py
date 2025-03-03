import pandas as pd
import numpy as np
from scipy.stats import zscore
import pickle
import os 
import duckdb

class DevicePreprocessor:
    def __init__(self, db_path, parquet_file, pkl_path,output_parquet_path,count_parameter, limit=1000000):
        self.db_path = db_path
        self.pkl_path = pkl_path
        self.output_parquet_path=output_parquet_path
        self.parquet_file = parquet_file
        self.limit = limit
        self.con = duckdb.connect(db_path)
        self.data = None
        self.filtered_data = None
        self.sw_category_counts = None
        self.valid_devices = None
        self.weekwise_data = None
        self.l1_distances = None
        self.count_parameter=count_parameter

    def load_data(self):
        """Load data from the Parquet file into a Pandas DataFrame."""
        query = f"SELECT interval_start_utc, proc_name, guid FROM '{self.parquet_file}' LIMIT {self.limit}"
        self.data = self.con.execute(query).fetchdf()

    def extract_year_week(self):
        """Extract year and week information from `interval_start_utc`."""
        self.data["year_week"] = self.data["interval_start_utc"].dt.strftime('%Y-%U')

    def count_cat_name_per_device(self):
        """Count the occurrences of `sw_category` per `guid`."""
        with open(self.pkl_path, 'rb') as file:
            sw_cat = pickle.load(file)
        self.data['sw_category'] = self.data['proc_name'].map(lambda x: sw_cat.get(x, "Other"))
        self.sw_category_counts = self.data.groupby(["guid", "year_week", "sw_category"]).size().reset_index(name="count")

    def filter_valid_devices(self):
        """Filter devices where `sw_category_name` count is greater than a threshold."""
        device_counts = self.sw_category_counts.groupby("guid")["count"].sum().reset_index()
        self.valid_devices = device_counts[device_counts["count"] > self.count_parameter]

    def merge_filtered_data(self):
        """Merge filtered devices with the original category count data."""
        self.filtered_data = self.sw_category_counts[self.sw_category_counts["guid"].isin(self.valid_devices["guid"])]

    def standardize_usage_counts(self):
        """Standardize category usage counts within each device using Z-score."""
        self.filtered_data["zscore_count"] = self.filtered_data.groupby("guid")["count"].transform(zscore)

    def compute_l1_distances(self):
        """Compute L1 distance for standardized category counts across consecutive weeks per device."""
        self.weekwise_data = self.filtered_data.pivot_table(index=["guid", "sw_category"], columns="year_week", values="zscore_count", aggfunc="sum").fillna(0)
        self.l1_distances = self.weekwise_data.diff(axis=1).abs().sum(axis=1).reset_index()
        self.l1_distances = self.l1_distances.groupby("guid")[0].sum().reset_index()
        self.l1_distances.columns = ['guid', 'l1_distance']

    def save(self):
        """Save the loaded data into a Parquet file."""
        if self.data is not None:
            if self.l1_distances is not None:
                self.l1_distances.to_parquet(self.output_parquet_path, index=False)
                print(f"Data saved to {self.output_parquet_path}")
        else:
            print("No data to save. Make sure to process the data first.")
    def preprocess(self):
        """Run all preprocessing steps."""
        self.load_data()
        self.extract_year_week()
        self.count_cat_name_per_device()
        self.filter_valid_devices()
        self.merge_filtered_data()
        self.standardize_usage_counts()
        self.compute_l1_distances()
        self.save()


