import src.LR_PVAL.src.process_data as process_data
import os
import json


def main(**params):
    # TODO: Implement the main function
    if "" in params:
        with open("config.json") as fh:
            params = json.load(fh)
            verbose = params.get("verbose")

            lr_pval_params = params.get("lr-pval-params")

            item_dir = lr_pval_params.get("item_dir")
            header_dir = lr_pval_params.get("header_dir")
            pq_output_dir = lr_pval_params.get("pq_output_dir")
            csv_output_dir = lr_pval_params.get("csv_output_dir")
            checkpoint_file = lr_pval_params.get("checkpoint_file")
            duck_temp_dir = lr_pval_params.get("duck_temp_dir")
    else:
        verbose = params.get("verbose")

        lr_pval_params = params.get("lr-pval-params")
        item_dir = lr_pval_params.get("item_dir")
        header_dir = lr_pval_params.get("header_dir")
        pq_output_dir = lr_pval_params.get("pq_output_dir")
        csv_output_dir = lr_pval_params.get("csv_output_dir")
        checkpoint_file = lr_pval_params.get("checkpoint_file")
        duck_temp_dir = lr_pval_params.get("duck_temp_dir")

    # Create directories if it doesn't exist
    os.makedirs(pq_output_dir, exist_ok=True)
    os.makedirs(csv_output_dir, exist_ok=True)
    os.makedirs(duck_temp_dir, exist_ok=True)

    # Make params
    process_data_params = {
        "item_dir": item_dir,
        "header_dir": header_dir,
        "pq_output_dir": pq_output_dir,
        "csv_output_dir": csv_output_dir,
        "checkpoint_file": checkpoint_file,
        "duck_temp_dir": duck_temp_dir,
        "verbose": verbose,
    }
    process_data.main(**process_data_params)


if __name__ == "__main__":
    with open("config.json") as fh:
        params = json.load(fh)

    main(**params)
