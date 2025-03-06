import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

from src.model_build import train
from src.feat_build import main

def main(targets):
    # read in parameters from command line
    if '' in targets:
        with open('config.json') as fh:
            data_params = json.load(fh)
            eps_vals = data_params["eps_vals"]
            data_fp = data_params.get("fp")
            output_fp = data_params.get("output")

            lasso_params = data_params["lasso-params"]
            
            l = lasso_params.get("l")
            tol = lasso_params.get("tol")
            max_iter = lasso_params.get("max_iter")
            c = lasso_params.get("c")
            directories = lasso_params.get("directories")
            sample_guids_parquet = lasso_params.get("sample_guids_parquet")
            model = lasso_params.get("model")
    else:
        eps_vals = None
        l = None
        tol = None
        max_iter = None
        c = None
        directories = None
        sample_guids_parquet = None
        model = None
        directories = None
        sample_guids_parquet = None
    # set default values if not specified
    if eps_vals is None: epss = [0.01, 0.1, 1, 10, 100]
    if l is None: l=10
    if tol is None: tol=1e-4
    if max_iter is None: max_iter=2500
    if c is None: c=0.1
    if model is None: model='fw-lasso-exp'
    if feat_parquet is None: feat_parquet = 'feat.parquet'
    if directories is None: directories = ["frgnd_backgrnd_apps_v4_hist", "web_cat_usage_v2","power_acdc_usage_v4_hist","os_c_state", "hw_pack_run_avg_pwr"]
    if sample_guids_parquet is None: sample_guids_parquet = 'sample_guids.parquet'

    inv_dir = Path(data_fp)
    #proj_dir = inv_dir.parent

    # create feat.parquet if it doesn't exist
    if feat_parquet not in os.listdir(data_fp / 'out'):
        main.generate_features(inv_dir / sample_guids_parquet, inv_dir, directories)
    else:
        print('Features already generated')

    # read data
    feat = pd.read_parquet(os.path.join(data_fp / 'out', feat_parquet))

    # run model on each epsilon value
    epsresults = []
    for eps in epss:
        test_mse, feat_dict, r2 = train.train(feat, model, tol=tol, l=l, epsilon=eps, max_iter=max_iter)
        
        epsresults.append(test_mse)
    
    # convert mse to utility
    rmses = np.sqrt(np.array(epsresults))
    max_rmse = np.max(rmses)
    # for higher values of c, punish rmse more. c in (0, inf)
    utility = 2 / (1 + np.exp(c * rmses / max_rmse)) # use sigmoid function to normalize
    
    # pretty task name for paper
    if model == 'fw-lasso-exp':
        model = 'LASSO'

    return pd.DataFrame({'task': [model for i in range(len(epss))], 
                         'epsilon': epss, 
                         'utility': utility.tolist()})

if __name__ == "__main__":
    main()