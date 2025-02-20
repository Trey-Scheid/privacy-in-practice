import pandas as pd
import numpy as np
from scipy.stats import zscore
import pickle
import os 
import duckdb
from src.utils import software_categories

class DevicePreprocessor:
    def __init__(self, db_path, parquet_file, pkl_path,output_parquet_path, limit=1000000):
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

    def load_data(self):
        """Load data from the Parquet file into a Pandas DataFrame."""
        query = f"SELECT interval_start_utc, proc_name, guid, duration FROM '{self.parquet_file}' LIMIT {self.limit}"
        self.data = self.con.execute(query).fetchdf()

    def extract_year_week(self):
        """Extract year and week information from `interval_start_utc`."""
        self.data["year"] = self.data["interval_start_utc"].dt.year
        self.data["week"] = self.data["interval_start_utc"].dt.isocalendar().week

    def count_cat_name_per_device(self):
        """Count the occurrences of `sw_category` per `guid`."""
        with open(self.pkl_path, 'rb') as file:
            sw_cat = pickle.load(file)
            self.data['sw_category'] = self.data['proc_name'].map(lambda x: sw_cat.get(x, "Other"))  # map software names to categories
        self.sw_category_counts = self.data.groupby("guid")['sw_category'].count().reset_index(name="sw_category_count")
 
    def filter_valid_devices(self):
        """Filter devices where `sw_category_name` count is greater than 10,000."""
        #10000 num switches as per paper (1000 for test)
        self.valid_devices = self.sw_category_counts[self.sw_category_counts["sw_category_count"] > 10000]

    def merge_filtered_data(self):
        """Merge filtered devices with the original data."""
        self.filtered_data = pd.merge(self.data, self.valid_devices[["guid"]], on="guid", how="inner")

    def standardize_duration(self):
        """Standardize the `duration` using Z-score per device."""
        self.filtered_data["zscore_duration"] = self.filtered_data.groupby("guid")["duration"].transform(zscore)

    def compute_l1_distances(self):
        """Compute L1 distance for consecutive weeks per device."""
        self.weekwise_data = self.filtered_data.pivot_table(index="guid", columns="week", values="zscore_duration", aggfunc="sum").fillna(0)
        self.l1_distances = self.weekwise_data.diff(axis=1).abs().sum(axis=1)
    def create_software_category_map(self):
        # Read the CSV file

        def categorize_process(name):
            name = name.lower()
            if pd.isna(name):
                return 'Unknown'
            
            for category, keywords in software_categories.items():
                if any(keyword in name for keyword in keywords):
                    return category
                    
            return 'Other'
        
        # Handle missing values and convert to string
        self.data['proc_name'] = self.data['proc_name'].fillna('').astype(str)

        self.data['sw_category'] = self.data['proc_name'].apply(categorize_process)

        # Create mapping dictionary and save to pickle
        software_mapping = self.data[['proc_name', 'sw_category']].set_index('proc_name').to_dict()['sw_category']

        # Save to pickle
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(software_mapping, f)
            print(f'Software category mapping saved to ' + str(self.pkl_path))
        f.close()
        return
    def pkl(self):
        #if no pkl
        
        # creates mapping file
        if not os.path.exists(self.pkl_path):
            print('Creating software category mapping pickle')
            self.create_software_category_map()
    def load(self):
        """Save the loaded data into a Parquet file."""
        if self.data is not None:
            df = self.l1_distances.reset_index()  # Convert Series to DataFrame
            df.columns = ['guid', 'l1_distance'] 
            df.to_parquet(self.output_parquet_path, index=False)
            print(f"Data saved to {self.output_parquet_path}")
        else:
            print("No data to save. Make sure to load the data first.")
    def preprocess(self):
        """Run all preprocessing steps."""
        self.load_data()
        self.extract_year_week()
        self.pkl()
        self.count_cat_name_per_device()
        self.filter_valid_devices()
        self.merge_filtered_data()
        self.standardize_duration()
        self.compute_l1_distances()
        self.load()

