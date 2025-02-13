import pyarrow as pa
import random


def sample_table(table: pa.Table, n_sample_rows: int = None) -> pa.Table:
    if n_sample_rows is None or n_sample_rows >= table.num_rows:
        return table

    indices = random.sample(range(table.num_rows), k=n_sample_rows)

    return table.take(indices)
