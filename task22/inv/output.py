import os, sys
import pandas as pd
from pathlib import Path

if os.path.exists(os.path.abspath('../src')):
    sys.path.append(os.path.abspath('../src'))

from feat_build import main
from model_build import train

inv_dir = Path(os.getcwd())
proj_dir = inv_dir.parent

sample_guids_parquet = 'sample_guid_10000_china_us.parquet'

if 'feat.parquet' not in os.listdir(inv_dir / 'out'):
    main.generate_features(sample_guids_parquet, inv_dir)