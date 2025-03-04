import os
import pandas as pd
from pathlib import Path
import sqlalchemy as sa
from feat_build.utils import table_names, url, src, global_data

# connect to the database
engine = sa.create_engine(url)

def dl_sysinfo():
    """
    downloads sysinfo table
    """
    #if sysinfo table doesn't exist, download from database
    print('Downloading sysinfo data from database...')

    (
        pd
        .read_sql_table('system_sysinfo_unique_normalized', engine, schema='university_analysis_pad')
        .to_parquet(global_data / 'sysinfo.parquet', index=False)
    )

def sample_raw(table_name, output_dir):
    """
    sample raw data given sample GUID table and saves it as parquet file in data/raw

    :param table_name: name of table with GUIDs for the sample (in the public schema)
    :param output_dir: directory to save the raw data files
    :return: True if successful
    """

    # read the queries for raw data
    with open(src / 'data' / 'raw_queries.sql', 'r') as file:
        queries = file.read()

    q_list = (
        queries
        .replace('sample_table', table_name)  # replace 'sample_table' (placeholder str) with table_name
        .split(';')  # split the queries into single queries
    )[:-1]  # has empty string in the end

    # execute each and read the results into .parquet files
    for i, query in enumerate(q_list):
        if (f'{table_names[i]}.parquet' not in os.listdir(output_dir)):
            print(f'Downloading {table_names[i]} data from database...')
            pd.read_sql_query(query, engine).to_parquet(output_dir / f'{table_names[i]}.parquet', index=False)

    return True
