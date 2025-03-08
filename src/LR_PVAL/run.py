import src.process_data as process_data
import os
import json


def main(targets):
    # TODO: Implement the main function
    if "" in targets:
        with open("config.json") as fh:
            params = json.load(fh)
            lr_pval_params = params.get("lr-pval-params")
            item_dir = lr_pval_params.get("item_dir")
            header_dir = lr_pval_params.get("header_dir")
            output_dir = lr_pval_params.get("output_dir")
            checkpoint_file = lr_pval_params.get("checkpoint_file")
            duck_temp_dir = lr_pval_params.get("duck_temp_dir")

    # Create directories if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(duck_temp_dir, exist_ok=True)

    process_data.main(item_dir, header_dir, output_dir, checkpoint_file, duck_temp_dir)


if __name__ == "__main__":
    with open("config.json") as fh:
        params = json.load(fh)

    main(**params)
