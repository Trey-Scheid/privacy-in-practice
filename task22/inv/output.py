import os, sys
import pandas as pd
from pathlib import Path

if os.path.exists(os.path.abspath('../src')):
    sys.path.append(os.path.abspath('../src'))

from feat_build import main
from model_build import train, frankWolfeLASSO

inv_dir = Path(os.getcwd())
proj_dir = inv_dir.parent

sample_guids_parquet = 'sample_guid_10000_china_us.parquet'

if 'feat.parquet' not in os.listdir(inv_dir / 'out'):
    main.generate_features(sample_guids_parquet, inv_dir)

feat = pd.read_parquet(os.path.join('out', 'feat.parquet'))

test_mse, feat_dict, r2 = train.train(feat, "lasso")
print("R2: ", r2)

test_mse, feat_dict, r2 = train.train(feat, "fw-lasso")
print("R2: ", r2)

# test_mse, feat_dict, r2 = train.train(pd.concat([feat.iloc[0:50,]["power_mean"], feat.drop("power_mean", axis=1).iloc[0:50, 1:3]], axis=1), "fw-lasso")
# print("R2: ", r2)
