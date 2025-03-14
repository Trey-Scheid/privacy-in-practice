import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

from src.LASSO.src.model_build import train
from src.LASSO.src.feat_build import main


def main(**data_params):
    # read in parameters from command line
    eps_vals = data_params["eps_vals"]
    data_fp = data_params.get("fp")
    output_fp = data_params.get("output")
    target_eps = data_params.get("single_epsilon")
    delta = data_params.get("delta")
    v = data_params.get("verbose")

    lasso_params = data_params["lasso-params"]
    
    l = lasso_params.get("l")
    tol = lasso_params.get("tol")
    max_iter = lasso_params.get("max_iter")
    c = lasso_params.get("c")
    directories = lasso_params.get("directories")
    sample_guids_parquet = lasso_params.get("sample_guids_parquet")
    model = lasso_params.get("model")

    # set default values if not specified
    if eps_vals is None: eps_vals = [0.01, 0.1, 1, 10, 100]
    if l is None: l=10
    if tol is None: tol=1e-4
    if max_iter is None: max_iter=2500
    if c is None: c=0.1
    if model is None: model='fw-lasso-exp'
    if feat_parquet is None: feat_parquet = 'feat.parquet'
    if directories is None: directories = ["frgnd_backgrnd_apps_v4_hist", "web_cat_usage_v2","power_acdc_usage_v4_hist","os_c_state", "hw_pack_run_avg_pwr"]
    if sample_guids_parquet is None: sample_guids_parquet = 'sample_guids.parquet'

    data_dir = Path(data_fp)
    processed_dir = data_dir / 'processed'
    raw_dir = data_dir / 'raw'

    # create feat.parquet if it doesn't exist
    if feat_parquet not in os.listdir(processed_dir):
        status = main.generate_features(raw_dir / sample_guids_parquet, raw_dir, processed_dir, directories)
        print(f"Features saved to {feat_parquet}") if status else print("unknown failure")
    else:
        print(f'{feat_parquet} being used as model features...')

    # read data
    feat = pd.read_parquet(processed_dir / feat_parquet)

    output_fp = output_fp / 'lasso'
    if not output_fp.exists():
        os.mkdir(output_fp)

    task_utility = train.train_run_eps(eps_vals, model, feat, tol, l, max_iter, plot_dir=output_fp, baseline=True)
    
    # train.research_plots()

    return task_utility


if __name__ == "__main__":
    with open("config/run.json") as fh:
        params = json.load(fh)

    main(**params)