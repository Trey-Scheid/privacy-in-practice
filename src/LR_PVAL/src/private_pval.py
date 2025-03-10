import pandas as pd
import numpy as np
from scipy import stats
import re
import glob
import os
import plotly.graph_objects as go
import argparse


def parse_private_results(
    data_dir: str,
    private_results_fp: str = "private_analysis.csv",
) -> pd.DataFrame:
    results = pd.read_csv(os.path.join(data_dir, private_results_fp))
    results = results.set_index("Unnamed: 0")
    results = results[["2"]]
    results.columns = ["value"]

    # First, let's reset the index to get the bugcheck_code as a column
    results = results.reset_index()
    results.columns = ["bugcheck_code", "value"]
    results = results[results["bugcheck_code"] != 0]
    # Create an empty DataFrame to store the parsed results
    parsed_results = []

    # Parse each row
    for _, row in results.iterrows():
        bugcheck_code = row["bugcheck_code"]
        # Replace numpy-specific strings with Python equivalents
        value_str = (
            row["value"].replace("np.float64", "float").replace("array", "np.array")
        )
        # Convert string representation of dict to actual dict
        data_dict = eval(value_str)

        # Extract data for each iteration
        for n_iter, values in data_dict.items():
            parsed_results.append(
                {
                    "bugcheck_code": bugcheck_code,
                    "n_iter": n_iter,
                    "epsilon": values["epsilon"],
                    "loss": float(
                        values["loss"]
                    ),  # Convert np.float64 to regular float
                    "param_0": float(
                        values["params"][0]
                    ),  # Convert numpy array elements to float
                    "param_1": float(values["params"][1]),
                }
            )

    # Convert to DataFrame
    results_df = pd.DataFrame(parsed_results)
    results_df["wald_statistic"] = results_df.apply(
        lambda row: logistic_regression_wald_test_from_params(
            get_X(row["bugcheck_code"], data_dir),
            row[["param_0", "param_1"]].to_list(),
        )["wald_statistic"],
        axis=1,
    )
    return results_df


def get_perm_df(bugcheck_id: int, data_dir: str):
    perm_dir = os.path.join(data_dir, "permutation_results")
    perm_df = pd.read_csv(os.path.join(perm_dir, f"{bugcheck_id}.csv"))
    perm_df = perm_df.unstack().reset_index()
    perm_df.columns = ["n_iter", "n_iter_idx", "value"]

    perm_df["epsilon"] = perm_df["value"].apply(
        lambda x: re.findall(r"'epsilon': (\d+\.\d+)", str(x))[0]
    )
    perm_df["loss"] = perm_df["value"].apply(
        lambda x: re.findall(r"'loss': np\.float64\((\d+\.\d+)\)", str(x))[0]
    )

    def extract_param(x, group_num):
        match = re.search(r"'params': array\(\[(-?\d+\.\d+).+ (-?\d+\.\d+)", str(x))
        return match.group(group_num) if match else None

    perm_df["param_0"] = perm_df["value"].apply(lambda x: extract_param(x, 1))
    perm_df["param_1"] = perm_df["value"].apply(lambda x: extract_param(x, 2))

    perm_df = perm_df.drop(columns=["value"])
    perm_df[["epsilon", "loss", "param_0", "param_1"]] = perm_df[
        ["epsilon", "loss", "param_0", "param_1"]
    ].astype(float)
    perm_df[["n_iter", "n_iter_idx"]] = perm_df[["n_iter", "n_iter_idx"]].astype(int)

    return perm_df


def get_X(bugcheck_id: int, data_dir: str):
    bugcheck_id = int(bugcheck_id)

    X = pd.read_csv(os.path.join(data_dir, f"bugcheck_{bugcheck_id}.csv"))
    X = X["has_corrected_error"]
    X = X.to_numpy()
    X = X.reshape(-1, 1)
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    return X


def logistic_regression_wald_test_from_params(X: np.ndarray, params: np.ndarray):
    """
    Perform Wald test for a logistic regression coefficient using only parameters and X.

    Args:
        X (np.array): Feature matrix
        params (np.array): Model parameters/coefficients
        param_index (int): Index of the parameter to test (default=0 for first feature)

    Returns:
        dict: Contains test statistic, p-value, and standard error
    """
    # Get coefficient we want to test
    beta = params[1]

    if not np.any(X[:, 1] == 1):
        return {
            "coefficient": beta,
            "std_error": 0,
            "wald_statistic": 0,
            "p_value": 1,
        }
    # Calculate predicted probabilities
    z = X @ params  # Linear combination
    p = 1 / (1 + np.exp(-z))  # Logistic function

    # Get unique p values and their indices
    p0 = p[X[:, 1] == 0][0]  # p value when X=0
    p1 = p[X[:, 1] == 1][0]  # p value when X=1
    n0 = np.sum(X[:, 1] == 0)  # count of X=0
    n1 = np.sum(X[:, 1] == 1)  # count of X=1

    # Compute X.T @ W @ X directly
    # This is a 2x2 matrix:
    # [sum(p*(1-p))        sum(p*(1-p)*x)]
    # [sum(p*(1-p)*x)      sum(p*(1-p)*x*x)]
    w00 = n0 * p0 * (1 - p0) + n1 * p1 * (1 - p1)
    w01 = n1 * p1 * (1 - p1)  # x is 1 for these terms
    w11 = n1 * p1 * (1 - p1)  # x*x is 1 for these terms (since x is binary)

    information_matrix = np.array([[w00, w01], [w01, w11]])

    # Calculate inverse of information matrix
    # For 2x2 matrix, we can do this directly
    det = (
        information_matrix[0, 0] * information_matrix[1, 1]
        - information_matrix[0, 1] ** 2
    )
    vcov = (
        1
        / det
        * np.array(
            [
                [information_matrix[1, 1], -information_matrix[0, 1]],
                [-information_matrix[0, 1], information_matrix[0, 0]],
            ]
        )
    )

    # Get standard error for the coefficient
    std_err = np.sqrt(vcov[1, 1])

    # Calculate Wald statistic
    wald_stat = (beta / std_err) ** 2

    # Calculate p-value (chi-square with 1 df)
    p_value = 1 - stats.chi2.cdf(wald_stat, df=1)

    return {
        "coefficient": beta,
        "std_error": std_err,
        "wald_statistic": wald_stat,
        "p_value": p_value,
    }


def get_percentile(val, arr):
    return np.sum(arr <= val) / len(arr)


def get_pval(val, arr):
    return 1 - get_percentile(val, arr)


def calculate_all_pvals(
    results_df: pd.DataFrame,
    data_dir: str,
    verbose: bool = False,
    perm_fp: str = "permutation_results",
):
    fps = glob.glob(os.path.join(data_dir, perm_fp, "*.csv"))
    bugcheck_ids = [int(fp.split("/")[-1].split(".")[0]) for fp in fps]

    results = {}

    for bugcheck_id in bugcheck_ids:
        try:
            perm_df = get_perm_df(bugcheck_id, data_dir)
            X = get_X(bugcheck_id, data_dir)
        except:
            raise Exception(f"Error getting perm_df for {bugcheck_id}")
        result_id = results_df[results_df["bugcheck_code"] == bugcheck_id]

        perm_df["wald_statistic"] = perm_df.apply(
            lambda row: logistic_regression_wald_test_from_params(
                X, row[["param_0", "param_1"]].to_list()
            )["wald_statistic"],
            axis=1,
        )
        perm_df = pd.merge(
            perm_df,
            result_id[["n_iter", "wald_statistic"]],
            on="n_iter",
            how="left",
            suffixes=("", "_original"),
        )

        results[bugcheck_id] = (
            perm_df.groupby("n_iter")
            .apply(
                lambda x: get_pval(
                    x["wald_statistic_original"].values[0], x["wald_statistic"]
                ),
                include_groups=False,
            )
            .to_dict()
        )
        if verbose:
            print(f"Completed {bugcheck_id}")

    significant_raw = pd.DataFrame(results).T
    significant_raw = significant_raw.unstack().reset_index()
    significant_raw.columns = ["n_iter", "bugcheck_code", "p_val"]
    significant_raw[["n_iter", "bugcheck_code"]] = significant_raw[
        ["n_iter", "bugcheck_code"]
    ].astype(int)
    significant_raw = pd.merge(
        significant_raw,
        results_df[["n_iter", "epsilon", "param_0", "param_1"]]
        .groupby("n_iter")
        .first(),
        on="n_iter",
        how="left",
    )
    significant = significant_raw[significant_raw["p_val"] < 0.05]

    significant.to_csv(
        os.path.join(data_dir, "private_analysis_pvals.csv"), index=False
    )

    return significant


def get_nonprivate_significant_set(
    data_dir: str,
    nonprivate_fp: str = "nonprivate_analysis.csv",
):
    nonprivate = pd.read_csv(os.path.join(data_dir, nonprivate_fp))
    nonprivate = nonprivate.rename(columns={"Unnamed: 0": "bugcheck_code"})
    nonprivate = nonprivate[nonprivate["bugcheck_code"] != 0]
    for bugcheck_code in nonprivate["bugcheck_code"].unique():
        n = pd.read_csv(os.path.join(data_dir, f"bugcheck_{bugcheck_code}.csv")).shape[
            0
        ]
        nonprivate.loc[nonprivate["bugcheck_code"] == bugcheck_code, "n"] = n
    nonprivate_set = set(
        nonprivate.loc[nonprivate["pvalue"] < 0.05, "bugcheck_code"].to_list()
    )
    return nonprivate_set


def get_private_significant_set(
    data_dir: str,
    significant: pd.DataFrame | None = None,
    pval_fp: str = "private_analysis_pvals.csv",
) -> pd.Series:
    if significant is None:
        significant = pd.read_csv(os.path.join(data_dir, pval_fp))

    return significant.groupby("n_iter")["bugcheck_code"].apply(
        lambda x: set(x.to_list())
    )


def intersection_over_union(set1, set2) -> float:
    return len(set1.intersection(set2)) / len(set1.union(set2))


def get_intersection_over_union_df(
    nonprivate_set: set,
    private_set: pd.Series,
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    iou = private_set.apply(
        lambda x: intersection_over_union(x, nonprivate_set)
    ).sort_index()
    iou = pd.DataFrame(iou).reset_index()
    iou.columns = ["n_iter", "iou"]
    iou = pd.merge(
        iou,
        results_df[["n_iter", "epsilon"]].groupby("n_iter").first(),
        on="n_iter",
        how="left",
    )

    return iou


def get_confusion_matrix(
    nonprivate_set: set, private_set: set, all_bugcheck_codes: set
) -> pd.DataFrame:
    # Create a confusion matrix for private_set[4096]
    ground_truth_significant = nonprivate_set
    ground_truth_not_significant = all_bugcheck_codes - ground_truth_significant
    private_significant = private_set
    private_not_significant = all_bugcheck_codes - private_significant

    # Create the confusion matrix
    true_positive = len(ground_truth_significant.intersection(private_significant))
    false_positive = len(private_significant) - true_positive
    false_negative = len(ground_truth_significant) - true_positive
    true_negative = len(
        ground_truth_not_significant.intersection(private_not_significant)
    )

    confusion_matrix = pd.DataFrame(
        [[true_positive, false_positive], [false_negative, true_negative]],
        index=["Predicted Significant", "Predicted Not Significant"],
        columns=["Significant", "Not Significant"],
    )

    print(confusion_matrix)

    return confusion_matrix


def plot_confusion_matrix(
    confusion_matrix: pd.DataFrame | np.ndarray,
    with_labels: bool = True,
    output_dir: str = "viz/static_output/LR_PVAL",
):
    if isinstance(confusion_matrix, pd.DataFrame):
        confusion_matrix = confusion_matrix.T.to_numpy()

    n_classes = confusion_matrix.shape[0]

    # Construct Sankey source-target pairs
    sources = []
    targets = []
    values = []

    for i in range(n_classes):  # Actual class
        for j in range(n_classes):  # Predicted class
            if confusion_matrix[i, j] > 0:
                sources.append(i)  # Actual class index
                targets.append(j + n_classes)  # Predicted class index
                values.append(confusion_matrix[i, j])

    # Calculate n values for each category
    n_sig_baseline = confusion_matrix[0].sum()  # Sum of first row
    n_not_sig_baseline = confusion_matrix[1].sum()  # Sum of second row
    n_sig_private = confusion_matrix[:, 0].sum()  # Sum of first column
    n_not_sig_private = confusion_matrix[:, 1].sum()  # Sum of second column

    # Create labels with n values
    all_labels = [
        f"Significant\n(n={n_sig_baseline})",
        f"Not Significant\n(n={n_not_sig_baseline})",
        f"Significant\n(n={n_sig_private})",
        f"Not Significant\n(n={n_not_sig_private})",
    ]

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="black", width=0.5),
                color=["#76ABAE", "#B2D0D2", "#76ABAE", "#B2D0D2"],
                label=all_labels if with_labels else None,
                # Position labels outside the nodes
            ),
            link=dict(source=sources, target=targets, value=values),
        )
    )

    fig.update_layout(
        title="Confusion Matrix as Sankey Diagram" if with_labels else None,
        font_size=12,
        width=1200,
        height=800,
        # Extend margins to accommodate outside labels
        margin=dict(l=200, r=200, t=150, b=80),
    )

    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(os.path.join(output_dir, "confusion_matrix.png"))

    return fig


def meta_analysis_csv(
    ious: pd.DataFrame | None = None,
    output_dir: str = "fiz/static_output/LR_PVAL",
):
    if ious is None:
        # TODO: run all code!!
        raise NotImplementedError

    metaanalysis_df = ious[["epsilon", "iou"]]
    metaanalysis_df["task"] = "LR_PVAL"
    metaanalysis_df.columns = ["epsilon", "utility", "task"]

    # metaanalysis_df.to_csv(os.path.join(output_dir, "lr_pval_meta.csv"), index=False)

    return metaanalysis_df


def eps_to_niter(epsilon: float, results_df: pd.DataFrame) -> int:
    """Find the iteration number that corresponds to the closest epsilon value in results_df.

    Args:
        epsilon: Target epsilon value
        results_df: DataFrame containing 'n_iter' and 'epsilon' columns

    Returns:
        The iteration number with the closest epsilon value
    """
    # Get unique epsilon-niter pairs
    eps_niter = results_df[["epsilon", "n_iter"]].drop_duplicates()

    # Find the closest epsilon value
    closest_eps = eps_niter.iloc[(eps_niter["epsilon"] - epsilon).abs().argsort()[0]]

    return int(closest_eps["n_iter"])


def main(
    data_dir: str,
    output_dir: str,
    epsilon: float = 2.0,
    verbose: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    results_df = parse_private_results(data_dir)
    significant = calculate_all_pvals(results_df, data_dir, verbose=verbose)
    nonprivate_set = get_nonprivate_significant_set(data_dir)
    private_set = get_private_significant_set(
        significant=significant, data_dir=data_dir
    )
    ious = get_intersection_over_union_df(nonprivate_set, private_set, results_df)

    try:
        private_set_vis = private_set.loc[eps_to_niter(epsilon, results_df)]
    except KeyError:
        private_set_vis = set()

    confusion_matrix = get_confusion_matrix(
        nonprivate_set,
        private_set_vis,
        set(results_df["bugcheck_code"].unique().tolist()),
    )

    plot_confusion_matrix(confusion_matrix, output_dir=output_dir)
    return meta_analysis_csv(ious)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, epsilon=1, verbose=True)
