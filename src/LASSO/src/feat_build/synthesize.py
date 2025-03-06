import os
from src.feat_build import load_data, process
from src.feat_build.utils import global_data

import pandas as pd
import numpy as np

def main(dummy_data_dir,  n_samples=1000):
    """
    Generate random noise following the schema provided in the feat_schema.csv file.
    The generated data is saved in the dummy_data_dir as synthetic_data.parquet.
    """
    schema = pd.read_csv(global_data / 'feat_schema.csv', columns=['column_names', 'data_type'])

    # Create random number generator
    rng = np.random.default_rng(seed=12345)  # Use seed for reproducibility

    # Generate all columns at once
    random_data = {
        col: (np.zeros(n_samples) if col in ['day_of_week', 'month_of_year'] 
            else rng.integers(0, 2, size=n_samples) if any(word in col for word in ["normalized_", "os_", "graphicsmanuf_", "cpu_", "persona_"]) else rng.random(size=n_samples)) # floats to 0-1
        for col, dtype in zip(schema['column_names'], schema['data_type'])
    }
    random_data["day_of_week"] = rng.integers(0, 7, size=n_samples)
    random_data["month_of_the_year"] = rng.integers(1, 13, size=n_samples)
    random_data["age_category"] = rng.integers(0, 10, size=n_samples)
    random_data["ram"] = rng.integers(0, 10, size=n_samples)
    random_data["#ofcores"] = rng.integers(0, 10, size=n_samples)
    random_data["screensize_category"] = rng.integers(0, 10, size=n_samples)
    
    random_df = pd.DataFrame(random_data)
    random_df.to_parquet(dummy_data_dir / 'Lasso_Regression'/ 'synthetic_data.parquet')

    return True