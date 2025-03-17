import pickle
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.LASSO.src.feat_build import sysinfo_process
from src.LASSO.src.feat_build.utils import software_categories

def create_software_category_map(output_dir, sw_raw, log=True):
    def categorize_process(name):
        name = name.lower()
        if pd.isna(name):
            return 'Unknown'
        for category, keywords in software_categories.items():
            if any(keyword in name for keyword in keywords):
                return category
        return 'Other'
    
    # Handle missing values and convert to string
    sw_raw['frgnd_proc_name'] = sw_raw['frgnd_proc_name'].fillna('').astype(str)

    sw_raw['Category'] = sw_raw['frgnd_proc_name'].apply(categorize_process)

    # Create mapping dictionary and save to pickle
    software_mapping = sw_raw[['frgnd_proc_name', 'Category']].set_index('frgnd_proc_name').to_dict()['Category']

    # Save to pickle
    with open(os.path.join(output_dir, 'software_data.pkl'), 'wb') as f:
        pickle.dump(software_mapping, f)
        if log:
            print(f'Software category mapping saved to {output_dir}/software_data.pkl')
    f.close()
    return

def process_raw(df, piv_val, cols=None, agg='sum'):
    """
    processes raw by pivoting the data and given columns

    :param df: raw data as pandas DataFrame
    :param piv_val: name of pivot value column
    :param cols: list of columns to pivot on
    :param agg: aggregation function as a string ('sum', 'mean', etc.) or list of strings
                if list, must be same length as cols

    :return: processed data as pandas DataFrame
    """
    if cols and not isinstance(cols, list):
        assert isinstance(cols, list), 'cols must be a list if passed in'

    if not cols:
        cols = df.columns.tolist()

    if isinstance(agg, list):
        assert len(agg) == len(cols), 'agg and cols must be same length'
        cols = list(zip(cols, agg))
    else:
        cols = list(zip(cols, [agg] * len(cols)))

    # store pivoted dataframes to concatenate later
    piv_df = []

    for col, agg_f in cols:
        piv = pd.pivot_table(
            df, 
            values=piv_val, 
            index=['guid', 'dt'], 
            columns=[col], 
            aggfunc=agg_f, 
            fill_value=0)
        
        # rename columns
        piv.columns = [f'{col}_{i}' for i in piv.columns]

        # append pivoted dataframe to list
        piv_df.append(piv)

    # concatenate pivoted dataframes
    proc_df = pd.concat(piv_df, axis=1).reset_index()
    
    # fill missing values with 0
    proc_df.fillna(0, inplace=True)
    
    return proc_df

def proc_temp(df):
    df['prod'] = (df['temp_nrs'] * df['avg_val']) / df.groupby(['guid', 'dt'])['temp_nrs'].transform('sum')

    return df.groupby(['guid', 'dt'])[['prod']].sum().reset_index().rename(columns={'prod': 'temp_avg'})

def main(raw_dir, proc_dir, proc_sysinfo=False, log=True):
    """
    main function to featureize non-sysinfo data
    
    :param data_folder: path to data folder for the investigation (not the raw data folder)
    :param proc_sysinfo: boolean to process sysinfo data, 
                         should be True only if sysinfo data is new
    """

    # load raw sample data
    if log:
        print('Loading raw data...')
    sw_raw = pd.read_parquet(proc_dir / 'sw_usage')  # software usage
    web_raw = pd.read_parquet(proc_dir / 'web_usage')  # web usage
    temp_raw = pd.read_parquet(proc_dir / 'temp')  # temperature
    cpu_raw = pd.read_parquet(proc_dir / 'cpu_util')  # temperature
    power_raw = pd.read_parquet(proc_dir / 'power')  # power (predictor variable)

    if not os.path.exists(proc_dir / 'software_data.pkl'):
        if log:
            print('Creating software category mapping pickle')
        # creates mapping file
        create_software_category_map(proc_dir, sw_raw, log)

    if log:
        print('Processing raw data...')

    # load software category data (mapping vocab from ChatGPT)
    with open(proc_dir / 'software_data.pkl', 'rb') as file:
        sw_cat = pickle.load(file)

    # process software usage data for pivoting
    sw_raw['sw_category'] = sw_raw['frgnd_proc_name'].map(sw_cat).fillna('Other')  # map software names to categories    
    sw_proc = process_raw(sw_raw, 'frgnd_proc_duration_ms', ['sw_category', 'sw_event_name'])
    web_proc = process_raw(web_raw, 'duration_ms', ['web_parent_category', 'web_sub_category'])
    temp_proc = proc_temp(temp_raw)

    # process CPU usage data
    cpu_raw.rename(columns={'norm_usage': 'cpu_norm_usage'}, inplace=True)

    # rename columns in power data
    power_raw.rename(columns={'mean': 'power_mean'}, inplace=True)
    power_raw.drop(columns='nrs_sum', inplace=True)

    # sysinfo one-hot encoding if not already done
    if 'sysinfo_ohe.parquet' not in os.listdir(proc_dir) or proc_sysinfo:
        if log:
            print('Processing sysinfo data...')
        sysinfo_process.main(raw_dir, proc_dir)
    
    sysinfo = pd.read_parquet(proc_dir / 'sysinfo_ohe.parquet')

    if log:
        print('Joining and standardizing data...')
    # merge dataframes (except power -- will merge last)
    merged_df = pd.merge(sw_proc, temp_proc, on=['guid', 'dt'], how='inner')
    merged_df = pd.merge(merged_df, web_proc, on=['guid', 'dt'], how='left')
    merged_df = pd.merge(merged_df, cpu_raw , on=['guid', 'dt'], how='inner')

    # standardize numerical columns
    scaler = StandardScaler()
    numeric_cols = merged_df.select_dtypes(include=['int', 'float']).columns.to_list()
    # if log:
        # print("Standardized:", numeric_cols)
        # print("Not Standardized:", set(merged_df.columns.to_list()).difference(set(numeric_cols)))
    merged_df[numeric_cols] = scaler.fit_transform(merged_df[numeric_cols])

    # don't standardize power columns
    merged_df = pd.merge(merged_df, power_raw, on=['guid', 'dt'], how='inner')

    # merge sysinfo data
    final_df = pd.merge(merged_df, sysinfo, on='guid', how='inner')
    final_df.fillna(0, inplace=True)

    # convert dt column to datetime
    final_df['dt'] = pd.to_datetime(final_df['dt'])

    # add time features
    final_df['day_of_week'] = final_df['dt'].dt.dayofweek
    final_df['month_of_year'] = final_df['dt'].dt.month

    # drop identifier columns (just features and target left)
    final_df.drop(columns=['dt', 'guid'], inplace=True)

    # save final featureized data (including target)
    if log:
        print('Writing features to disk...')
    final_df.to_parquet(proc_dir / 'feat.parquet')
    if log:
        print(f'Feature processing complete! feat.parquet saved to {proc_dir}')