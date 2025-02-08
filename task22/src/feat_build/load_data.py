import pandas as pd
import os
#from pathlib import Path
import duckdb
import pyarrow.parquet as pq
from feat_build.utils import table_names, global_data

def condense_parquet_files(input_directory, output_file):
    """Use if you have multiple parquet files in a directory that you want to combine into a single file, recommended for 16GB RAM to limit total parquet sizes to 5GB or less before condensing"""

    temp_table = pq.ParquetDataset(input_directory).read()
    pq.write_table(temp_table, output_file)

    print(f"Success, compressed {input_directory} into {output_file}")


def sample_raw(sample_guids_parquet, output_dir):
    con = duckdb.connect()
    
    # Create views for the parquet files
    views_sql = f"""
        CREATE VIEW frgnd_backgrnd_apps AS 
            SELECT * FROM read_parquet('{os.path.join(global_data, "frgnd_backgrnd_apps_v4_hist", "[!.]*.parquet").replace(os.sep, "/")}');
        CREATE VIEW web_cat_usage AS 
            SELECT * FROM read_parquet('{os.path.join(global_data, "web_cat_usage_v2", "[!.]*.parquet").replace(os.sep, "/")}');
        CREATE VIEW power_acdc_usage AS 
            SELECT * FROM read_parquet('{os.path.join(global_data, "power_acdc_usage_v4_hist", "[!.]*.parquet").replace(os.sep, "/")}');
        CREATE VIEW os_c_state AS 
            SELECT * FROM read_parquet('{os.path.join(global_data, "os_c_state", "[!.]*.parquet").replace(os.sep, "/")}');
        CREATE VIEW hw_pack_run_avg_pwr AS 
            SELECT * FROM read_parquet('{os.path.join(global_data, "hw_pack_run_avg_pwr", "[!.]*.parquet").replace(os.sep, "/")}');
        CREATE VIEW sample_guids AS 
            SELECT * FROM read_parquet('{os.path.join(global_data, sample_guids_parquet,).replace(os.sep, "/")}');
    """

    con.execute("""-- Set memory limits before running query
                    SET memory_limit='16GB';
                    -- Enable progress tracking
                    SET enable_progress_bar=true;
                    -- Enable detailed profiling
                    SET profiling_mode='detailed';
                """)
    con.execute(views_sql)


    queries = {        
        'web_usage': """
                SELECT sg.guid, wb.dt, wb.browser, 
                       wb.parent_category AS web_parent_category, 
                       wb.sub_category AS web_sub_category, wb.duration_ms
                FROM sample_guids sg
                JOIN web_cat_usage wb ON sg.guid = wb.guid
        """,

        'sw_usage': """
                SELECT sg.guid, sw.event_name AS sw_event_name, 
                       sw.frgnd_proc_dt AS dt, sw.frgnd_proc_name, 
                       sw.frgnd_proc_duration_ms
                FROM sample_guids sg
                JOIN frgnd_backgrnd_apps sw ON sg.guid = sw.guid
        """,
        
        'temp': """
                WITH pwr AS (
                    SELECT guid, dt, event_name AS temp_event_name, 
                           duration_ms, metric_name, 
                           attribute_metric_level1 AS temp_attribute_metric_level1, 
                           nrs, avg_val
                    FROM power_acdc_usage
                    WHERE metric_name LIKE '%TEMPERATURE%'
                )
                SELECT pwr.*
                FROM pwr
                JOIN sample_guids sg ON pwr.guid = sg.guid
        """,
        
        'cpu_util': """
                WITH core_count AS (
                    SELECT c.guid, CAST(MAX(cpu_id) AS int) + 1 AS cores
                    FROM os_c_state c
                    JOIN sample_guids s ON c.guid = s.guid
                    WHERE c.cpu_id != '_TOTAL'
                    GROUP BY c.guid
                ),
                cpu_usage AS (
                    SELECT c.guid, dt, 
                           SUM(sample_count * average) / SUM(sample_count) AS usage, 
                           SUM(sample_count) AS nrs
                    FROM os_c_state c
                    JOIN sample_guids s ON c.guid = s.guid
                    WHERE c.cpu_id = '_TOTAL'
                    GROUP BY c.guid, dt
                )
                SELECT cu.guid, cu.dt, cc.cores * cu.usage AS norm_usage, nrs
                FROM core_count cc
                JOIN cpu_usage cu ON cc.guid = cu.guid
        """,
        
        'power': """
                SELECT pw.guid, pw.dt, 
                       SUM(pw.nrs * pw.mean) / SUM(pw.nrs) AS mean, 
                       SUM(pw.nrs) AS nrs_sum
                FROM sample_guids sg
                JOIN hw_pack_run_avg_pwr pw ON sg.guid = pw.guid
                GROUP BY pw.guid, pw.dt
        """
    }

    for table_name, query in queries.items():
        # output_file = output_dir / f'{table_name}.parquet'
        table_dir = output_dir / table_name
        if not table_dir.exists():
            table_dir.mkdir()
            print(f'Processing {table_name} data...')
            try:
                copy_query = f"""
                    COPY (
                        {query}
                    ) TO '{table_dir}' (FORMAT PARQUET, PER_THREAD_OUTPUT true, ROW_GROUP_SIZE 100000)
                """ #FILENAME_PATTERN 'part_{{uuid}}'
                con.execute(copy_query)
            except Exception as e:
                print(f'Error processing {table_name}: {str(e)}')
        else:
            print(f"{table_name} exists...")
    
    con.close()
    return True
