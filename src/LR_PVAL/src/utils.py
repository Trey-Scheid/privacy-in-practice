import pyarrow as pa
import random
import glob
import os


def sample_table(table: pa.Table, n_sample_rows: int = None) -> pa.Table:
    if n_sample_rows is None or n_sample_rows >= table.num_rows:
        return table

    indices = random.sample(range(table.num_rows), k=n_sample_rows)

    return table.take(indices)


def get_data_fps(data_dir: str) -> list[str]:
    data_fps = glob.glob(os.path.join(data_dir, "*.csv"))
    data_fps = [data_fp for data_fp in data_fps if "bugcheck" in data_fp]
    return data_fps
