import os
from feat_build import load_data, process
from feat_build.utils import global_data

def generate_features(sample_table, inv_data_dir):
    """
    generate features for a given sample table

    :param sample_table: name of table with GUIDs for the sample (in the public schema)
    :param inv_data_dir: data directory for the investigation
    :return: True if successful
    """

    # download sysinfo data if it doesn't exist
    if 'sysinfo.parquet' not in os.listdir(global_data):
        load_data.dl_sysinfo()

    raw_data_dir = inv_data_dir / 'raw'

    # download raw data for chosen sample table
    load_data.sample_raw(sample_table, raw_data_dir)

    # process raw data
    process.main(inv_data_dir)

    return True