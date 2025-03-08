import src.LR_PVAL.src.process_data as process_data
import src.LR_PVAL.src.nonprivate_analysis as nonprivate_analysis
import src.LR_PVAL.src.private_analysis as private_analysis
import src.LR_PVAL.src.private_pval as private_pval
import os
import json


def main(**params):
    # TODO: Implement the main function
    if "" in params:
        with open("config.json") as fh:
            params = json.load(fh)

    lr_pval_params = params.get("lr-pval-params")
    verbose = params.get("verbose")

    # Create directories if it doesn't exist
    os.makedirs(lr_pval_params.get("pq_output_dir"), exist_ok=True)
    os.makedirs(lr_pval_params.get("csv_output_dir"), exist_ok=True)
    os.makedirs(lr_pval_params.get("duck_temp_dir"), exist_ok=True)

    # Process data
    if verbose:
        print("Processing data")

    process_data_params = {
        "item_dir": lr_pval_params.get("item_dir"),
        "header_dir": lr_pval_params.get("header_dir"),
        "pq_output_dir": lr_pval_params.get("pq_output_dir"),
        "csv_output_dir": lr_pval_params.get("csv_output_dir"),
        "checkpoint_file": lr_pval_params.get("checkpoint_file"),
        "duck_temp_dir": lr_pval_params.get("duck_temp_dir"),
        "verbose": verbose,
    }

    process_data.main(**process_data_params)

    if verbose:
        print("Data processed, running non-private analysis")

    # Non-private analysis
    nonprivate_params = {
        "data_dir": lr_pval_params.get("csv_output_dir"),
    }
    nonprivate_analysis.main(**nonprivate_params)

    if verbose:
        print("Non-private analysis complete, running private analysis")

    # Private analysis
    private_analysis_params = {
        "data_dir": lr_pval_params.get("csv_output_dir"),
        "epsilon": lr_pval_params.get("max_epsilon"),
        "delta": params.get("delta"),
        "verbose": verbose,
        "log_gap": lr_pval_params.get("log_gap"),
        "mid_results": lr_pval_params.get("mid_results"),
    }

    private_analysis.get_all_private_lr_results(**private_analysis_params)

    private_permutation_params = {
        "epsilon": lr_pval_params.get("max_epsilon"),
        "delta": params.get("delta"),
        "data_dir": lr_pval_params.get("csv_output_dir"),
        "log_gap": lr_pval_params.get("log_gap"),
        "mid_results": lr_pval_params.get("mid_results"),
        "n_permutations": lr_pval_params.get("n_permutations"),
        "n_workers": lr_pval_params.get("n_workers"),
    }

    private_analysis.get_permutation_results(**private_permutation_params)

    private_pval_params = {
        "data_dir": lr_pval_params.get("csv_output_dir"),
        "output_dir": os.path.join(*params.get("output"), "LR_PVAL"),
        "epsilon": params.get("single_epsilon"),
        "verbose": verbose,
    }

    out = private_pval.main(**private_pval_params)

    print("Private analysis complete")

    return out


if __name__ == "__main__":
    with open("config/run.json") as fh:
        params = json.load(fh)

    main(**params)
