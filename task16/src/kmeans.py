import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.cluster import KMeans
import duckdb
"""
TODO:
- [ ] Add unit tests for the DeviceUsage class and its methods.
- [ ] group proc_names into categories
- [ ] Optimize the function and try to increase scope of the data (include more parquets/remove limit)
"""
class DeviceUsage:
    def __init__(self, db_path, parquet_file, limit=1000000):
        self.db_path = db_path
        self.parquet_file = parquet_file
        self.limit = limit
        self.con = duckdb.connect(db_path)
        self.data = None
        self.filtered_data = None
        self.proc_name_counts = None
        self.valid_devices = None
        self.weekwise_data = None
        self.l1_distances = None
        self.clusters = None

    def load_data(self):
        """Load data from the Parquet file into a Pandas DataFrame."""
        query = f"SELECT interval_start_utc,proc_name,guid,duration FROM '{self.parquet_file}' LIMIT {self.limit}"
        self.data = self.con.execute(query).fetchdf()

    def extract_year_week(self):
        """Extract year and week information from `interval_start_utc`."""
        self.data["year"] = self.data["interval_start_utc"].dt.year
        self.data["week"] = self.data["interval_start_utc"].dt.isocalendar().week

    def count_proc_name_per_device(self):
        """Count the occurrences of `proc_name` per `guid`."""
        self.proc_name_counts = self.data.groupby("guid")["proc_name"].count().reset_index(name="proc_name_count")

    def filter_valid_devices(self):
        """Filter devices where `proc_name` count is greater than 10,000."""
        self.valid_devices = self.proc_name_counts[self.proc_name_counts["proc_name_count"] > 10000]

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

    def apply_kmeans_clustering(self, num_clusters=3):
        """Apply K-Means clustering on L1 distances to group devices."""
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        self.clusters = kmeans.fit_predict(self.l1_distances.values.reshape(-1, 1))

    def get_clustered_devices(self):
        """Return the clustered device data."""
        return pd.DataFrame({
            "device_id": self.l1_distances.index,
            "L1_distance": self.l1_distances.values,
            "cluster": self.clusters
        })

    def run_analysis(self):
        """Run the full analysis pipeline."""
        self.load_data()
        self.extract_year_week()
        self.count_proc_name_per_device()
        self.filter_valid_devices()
        self.merge_filtered_data()
        self.standardize_duration()
        self.compute_l1_distances()
        self.apply_kmeans_clustering()
        return self.get_clustered_devices()

