import pandas as pd
import os
#from pathlib import Path
import duckdb
import pyarrow.parquet as pq
from src.LASSO.src.feat_build.utils import table_names

def condense_parquet_files(input_directory, output_file):
    """Use if you have multiple parquet files in a directory that you want to combine into a single file, recommended for 16GB RAM to limit total parquet sizes to 5GB or less before condensing"""

    temp_table = pq.ParquetDataset(input_directory).read()
    pq.write_table(temp_table, output_file)

    print(f"Success, compressed {input_directory} into {output_file}")


def sample_raw(raw_dir, sample_guids_parquet, output_dir, directories, per_thread=True, row_group_size=100000):
    con = duckdb.connect()
    con.execute("""-- Set memory limits before running query
                    SET memory_limit='16GB';
                    -- Enable progress tracking
                    SET enable_progress_bar=true;
                    -- Enable detailed profiling
                    SET profiling_mode='detailed';
                """)
    assert len(table_names) == len(directories)
    view_to_raw_data_directory_mapping = {(table_names[i] + "_view"):directories[i] for i in range(len(table_names))}

    # Create views for the parquet files
    create_view = """
        CREATE VIEW {view_name} AS 
            SELECT * FROM read_parquet('{parquet_path}');
    """
    
    queries = [
        # web_usage   
        """
            SELECT sg.guid, wb.dt, wb.browser, 
                    wb.parent_category AS web_parent_category, 
                    wb.sub_category AS web_sub_category, wb.duration_ms
            FROM sample_guids sg
            JOIN {table_name} wb ON sg.guid = wb.guid
        """,
        # sw_usage
        """
            SELECT sg.guid, sw.event_name AS sw_event_name, 
                    sw.frgnd_proc_dt AS dt, sw.frgnd_proc_name, 
                    sw.frgnd_proc_duration_ms
            FROM sample_guids sg
            JOIN {table_name} sw ON sg.guid = sw.guid
        """,
        # temp
        """
            WITH pwr AS (
                SELECT guid, dt, event_name AS temp_event_name, 
                        duration_ms, metric_name, 
                        attribute_metric_level1 AS temp_attribute_metric_level1, 
                        nrs, avg_val
                FROM {table_name}
                WHERE metric_name LIKE '%TEMPERATURE%'
            )
            SELECT pwr.*
            FROM pwr
            JOIN sample_guids sg ON pwr.guid = sg.guid
        """,
        # cpu_util
        """
            WITH core_count AS (
                SELECT c.guid, CAST(MAX(cpu_id) AS int) + 1 AS cores
                FROM {table_name} c
                JOIN sample_guids s ON c.guid = s.guid
                WHERE c.cpu_id != '_TOTAL'
                GROUP BY c.guid
            ),
            cpu_usage AS (
                SELECT c.guid, dt, 
                        SUM(sample_count * average) / SUM(sample_count) AS usage, 
                        SUM(sample_count) AS nrs
                FROM {table_name} c
                JOIN sample_guids s ON c.guid = s.guid
                WHERE c.cpu_id = '_TOTAL'
                GROUP BY c.guid, dt
            )
            SELECT cu.guid, cu.dt, cc.cores * cu.usage AS norm_usage, nrs
            FROM core_count cc
            JOIN cpu_usage cu ON cc.guid = cu.guid
        """,
        # power
        """
            SELECT pw.guid, pw.dt, 
                    SUM(pw.nrs * pw.mean) / SUM(pw.nrs) AS mean, 
                    SUM(pw.nrs) AS nrs_sum
            FROM sample_guids sg
            JOIN {table_name} pw ON sg.guid = pw.guid
            GROUP BY pw.guid, pw.dt
        """
    ]
    if any([not (output_dir / table_name).exists() for table_name in table_names]):
        # create sample_guid_view if any tables are missing
        parquet_path = os.path.join(raw_dir, sample_guids_parquet).replace(os.sep, "/")
        con.execute(create_view.format(view_name="sample_guids", parquet_path=parquet_path))

    for i, query in enumerate(queries):
        # output_file = output_dir / f'{table_name}.parquet'
        table_dir = output_dir / table_names[i]
        if not table_dir.exists():
            table_dir.mkdir()
            print(f'Processing {table_names[i]} data...')
            # create table view
            try:
                view_name = table_names[i] + "_view"
                parquet_path = os.path.join(raw_dir, view_to_raw_data_directory_mapping[view_name], "[!.]*.parquet").replace(os.sep, "/")
                con.execute(create_view.format(view_name=view_name, parquet_path=parquet_path))
                try:
                    copy_query = f"""
                        COPY (
                            {query}
                        ) TO '{table_dir}' (FORMAT PARQUET, PER_THREAD_OUTPUT {per_thread}, ROW_GROUP_SIZE {row_group_size})
                    """ #FILENAME_PATTERN 'part_{{uuid}}'
                    con.execute(copy_query)
                except Exception as e:
                    print(f'Error processing {table_names[i]}: {str(e)}')
            except Exception as e:
                    print(f'Error reading parquets for {table_names[i]}: {str(e)}')
        else:
            print(f"{table_names[i]} exists...")
    
    con.close()
    return True
