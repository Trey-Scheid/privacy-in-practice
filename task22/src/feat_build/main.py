import os
from feat_build import load_data, process
from feat_build.utils import global_data

def generate_features(sample_guids_parquet, inv_data_dir):
    """
    generate features for a given sample table

    :param sample_table: name of table with GUIDs for the sample (in the public schema)
    :param inv_data_dir: data directory for the investigation
    :return: True if successful
    """
    sysinfo_new = False
    # combine + rename sysinfo data if it doesn't exist
    if 'sysinfo.parquet' not in os.listdir(global_data):
        load_data.condense_parquet_files(global_data / "system_sysinfo_unique_normalized", global_data / 'sysinfo.parquet')
        sysinfo_new = True

    if sample_guids_parquet not in os.listdir(global_data):
        load_data.condense_parquet_files(global_data / sample_guids_parquet.replace(".parquet", ""), global_data / f'{sample_guids_parquet}.parquet')

    raw_data_dir = inv_data_dir / 'raw'

    # download raw data for chosen sample table
    load_data.sample_raw(sample_guids_parquet, raw_data_dir)

    # process raw data
    process.main(inv_data_dir, proc_sysinfo=sysinfo_new)

    return True