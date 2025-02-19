import pandas as pd
import statsmodels.api as sm
import os
import glob
import re

data_dir = "private_data/"
data_dir = os.path.abspath(data_dir)
data_fps = glob.glob(os.path.join(data_dir, "*.csv"))
data_fps = [data_fp for data_fp in data_fps if "bugcheck" in data_fp]


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
        "llf": result.llf,
    }


def get_all_lr_results():
    results = {
        int(re.findall(r"bugcheck_(\d+)", data_fp)[0]): get_lr_results(data_fp)
        for data_fp in data_fps
    }

    pd.DataFrame(results).T.to_csv(os.path.join(data_dir, "nonprivate_analysis.csv"))
    return results


if __name__ == "__main__":
    get_all_lr_results()
