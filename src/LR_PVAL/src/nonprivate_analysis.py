import pandas as pd
import statsmodels.api as sm
import os
import re
import json
from src.LR_PVAL.src.utils import get_data_fps


def get_lr_results(data_fp):
    df = pd.read_csv(data_fp)

    if df.groupby(["has_corrected_error", "has_bugcheck"]).size().shape[0] != 4:
        df = pd.concat(
            [df, pd.DataFrame({"has_corrected_error": [1], "has_bugcheck": [0]})],
            ignore_index=True,
        )
        df = pd.concat(
            [df, pd.DataFrame({"has_corrected_error": [1], "has_bugcheck": [1]})],
            ignore_index=True,
        )

    X = df[["has_corrected_error"]]
    X = sm.add_constant(X)
    y = df["has_bugcheck"]

    model = sm.Logit(y, X)
    result = model.fit()
    return {
        "pvalue": result.pvalues.iloc[1],
        "coef": result.params.iloc[1],
        "loss": -result.llf / len(y),
    }


def get_all_lr_results(data_dir):
    data_dir = os.path.abspath(data_dir)
    data_fps = get_data_fps(data_dir)

    results = {
        int(re.findall(r"bugcheck_(\d+)", data_fp)[0]): get_lr_results(data_fp)
        for data_fp in data_fps
    }

    pd.DataFrame(results).T.to_csv(os.path.join(data_dir, "nonprivate_analysis.csv"))
    return results


def main(data_dir):
    get_all_lr_results(data_dir)


if __name__ == "__main__":
    with open("config/run.json") as fh:
        params = json.load(fh)

    data_dir = params.get("lr-pval-params").get("csv_output_dir")
    main(data_dir)
