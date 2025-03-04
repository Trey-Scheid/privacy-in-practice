import pickle
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feat_build import sysinfo_process
from feat_build.utils import global_data, software_categories

def create_software_category_map(sw_raw):
    # Read the CSV file

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
    with open(os.join(global_data, 'software_data.pkl'), 'wb') as f:
        pickle.dump(software_mapping, f)
        print(f'Software category mapping saved to {global_data}/software_data.pkl')
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
    df['prod'] = (df['nrs'] * df['avg_val']) / df.groupby(['guid', 'dt'])['nrs'].transform('sum')

    return df.groupby(['guid', 'dt'])[['prod']].sum().reset_index().rename(columns={'prod': 'temp_avg'})

def main(data_folder, proc_sysinfo=False):
    """
    main function to featureize non-sysinfo data
    
    :param data_folder: path to data folder for the investigation (not the global data folder)
    :param proc_sysinfo: boolean to process sysinfo data, 
                         should be True only if sysinfo data is new
    """

    raw_folder = data_folder / 'raw'

    # load raw sample data
    print('Loading raw data...')
    sw_raw = pd.read_parquet(raw_folder / 'sw_usage')  # software usage
    web_raw = pd.read_parquet(raw_folder / 'web_usage')  # web usage
    temp_raw = pd.read_parquet(raw_folder / 'temp')  # temperature
    cpu_raw = pd.read_parquet(raw_folder / 'cpu_util')  # temperature
    power_raw = pd.read_parquet(raw_folder / 'power')  # power (predictor variable)

    if not os.path.exists(global_data / 'software_data.pkl'):
        print('Creating software category mapping pickle')
        # creates mapping file
        create_software_category_map(sw_raw)

    print('Processing raw data...')

    # load software category data (mapping vocab from ChatGPT)
    with open(global_data / 'software_data.pkl', 'rb') as file:
        sw_cat = pickle.load(file)

    # process software usage data for pivoting
    sw_raw['sw_category'] = sw_raw['frgnd_proc_name'].map(sw_cat)  # map software names to categories
    sw_raw['sw_category'] = sw_raw['sw_category'].fillna('Other')  # fill missing values with 'Other'
    
    sw_proc = process_raw(sw_raw, 'frgnd_proc_duration_ms', ['sw_category', 'sw_event_name'])
    web_proc = process_raw(web_raw, 'duration_ms', ['web_parent_category', 'web_sub_category'])
    temp_proc = proc_temp(temp_raw)

    # process CPU usage data
    cpu_raw.rename(columns={'norm_usage': 'cpu_norm_usage'}, inplace=True)

    # rename columns in power data
    power_raw.rename(columns={'mean': 'power_mean', 'nrs_sum': 'power_nrs_sum'}, inplace=True)
    power_raw.drop(columns='power_nrs_sum', inplace=True)

    # sysinfo one-hot encoding if not already done
    if 'sysinfo_ohe.parquet' not in os.listdir(global_data) or proc_sysinfo:
        print('Processing sysinfo data...')
        sysinfo_process.main()
        
    sysinfo = pd.read_parquet(global_data / 'sysinfo_ohe.parquet')

    print('Joining and standardizing data...')
    # merge dataframes (except power -- will merge last)
    merged_df = pd.merge(sw_proc, temp_proc, on=['guid', 'dt'], how='inner')
    merged_df = pd.merge(merged_df, web_proc, on=['guid', 'dt'], how='left')
    merged_df = pd.merge(merged_df, cpu_raw , on=['guid', 'dt'], how='inner')

    # standardize numerical columns
    scaler = StandardScaler()
    numeric_cols = merged_df.select_dtypes(include=['int', 'float']).columns.to_list()
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
    print('Writing features to disk...')
    final_df.to_parquet(data_folder / 'out' / 'feat.parquet')
    print(f'Feature processing complete! feat.parquet saved to {data_folder / "out"}')