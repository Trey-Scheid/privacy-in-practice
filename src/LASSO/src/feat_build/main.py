import os
from src.feat_build import load_data, process, synthesize
# from src.feat_build.utils import global_data

def generate_features(sample_guids_parquet, inv_data_dir, proc_data_dir, directories):
    """
    generate features for a given sample table

    :param sample_table: name of table with GUIDs for the sample (in the public schema)
    :param inv_data_dir: data directory for the investigation
    :return: True if successful
    """
    sysinfo_new = False
    # combine + rename sysinfo data if it doesn't exist
    if 'sysinfo.parquet' not in os.listdir(proc_data_dir):
        load_data.condense_parquet_files(inv_data_dir / "system_sysinfo_unique_normalized", proc_data_dir / 'sysinfo.parquet')
        sysinfo_new = True

    if sample_guids_parquet not in os.listdir(inv_data_dir):
        load_data.condense_parquet_files(inv_data_dir / sample_guids_parquet.replace(".parquet", ""), proc_data_dir / f'{sample_guids_parquet}.parquet')

    # download raw data for chosen sample table
    load_data.sample_raw(inv_data_dir, sample_guids_parquet, proc_data_dir, directories)

    # process raw data
    process.main(inv_data_dir, proc_data_dir, proc_sysinfo=sysinfo_new)

    return True

def generate_synthetic_data(dummy_data_dir):
    if 'feat.parquet' not in os.listdir(dummy_data_dir):
        synthesize.main(dummy_data_dir)
        return True

    print("feat.parquet already exists in the data directory. Skipping synthesis.")
    return False