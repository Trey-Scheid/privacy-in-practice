import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.LASSO.src.feat_build.utils import sysinfo_cols, screensize_mapping, age_cat, portable


def main(raw, proc):
    """
    process sysinfo data to merge with other tables (create sysinfo_ohe.parquet and chastype.parquet)
    """
    sysinfo_file = proc / 'sysinfo.parquet'

    # Load raw sysinfo data
    full_sysinfo = pd.read_parquet(sysinfo_file)

    # Split full_sysinfo into sysinfo (features) and chastype (chassistype)
    # Chassis Type is used to filter out guids later
    sysinfo, chastype = full_sysinfo[sysinfo_cols], full_sysinfo[['guid', 'chassistype']]

    # Fill missing values (n/a, N/A) with 'Unknown'
    sysinfo.replace({'n/a': 'Unknown', 'N/A': 'Unknown'}, inplace=True)

    # Leave only the Core-x suffix
    sysinfo['cpu_suffix'] = sysinfo['cpu_suffix'].apply(lambda x: x if 'Core' in x else 'Other')

    # Clean screensize_category
    sysinfo['screensize_category'] = sysinfo['screensize_category'].str.replace('x', '').replace(screensize_mapping)

    # Categorical columns (will be one-hot encoded)
    cat_cols = [
        'countryname_normalized',
        'modelvendor_normalized',
        'os',
        'graphicsmanuf',
        'cpu_family',
        'cpu_suffix',
        'persona'
    ]

    # Ordinal columns
    ord_cols = ['age_category']

    # Numerical columns
    num_cols = ['ram', '#ofcores', 'screensize_category']

    # Guid
    guid = ['guid']

    # Encoders and transformers
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ord = OrdinalEncoder(categories=[age_cat])

    cat_pip = Pipeline([
        ('one-hot', ohe)
    ])

    ord_pip = Pipeline([
        ('ordinal', ord)
    ])

    num_pip = Pipeline([
        ('impute', SimpleImputer(missing_values='Unknown', strategy='most_frequent')),
        ('standard', StandardScaler())
    ])

    proc = ColumnTransformer([
        ('categorical', cat_pip, cat_cols),
        ('ordinal', ord_pip, ord_cols),
        ('numerical', num_pip, num_cols),
    ], remainder='passthrough')

    sysinfo_proc = proc.fit_transform(sysinfo)

    ohe_cols = proc.named_transformers_['categorical'].named_steps['one-hot'].get_feature_names_out(cat_cols)

    all_cols = np.concatenate([ohe_cols, ord_cols, num_cols, guid])

    # Save processed sysinfo data
    proc_file_name = proc / 'sysinfo_ohe.parquet'
    pd.DataFrame(sysinfo_proc, columns=all_cols).to_parquet(proc_file_name, index=False)

    # Save chastype
    # Create column for portable (1) or desktop/server (0)
    chastype['portable'] = chastype['chassistype'].isin(portable).astype(int)

    chastype_file_name = proc / 'chastype.parquet'

    chastype.to_parquet(chastype_file_name, index=False)

if __name__ == '__main__':
    main()