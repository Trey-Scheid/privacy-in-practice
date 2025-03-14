#!/usr/bin/env python

import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

tasks = {}
try:
    from src.LASSO import run as runLasso
    tasks["lasso"] = runLasso
except (ImportError, ModuleNotFoundError) as e:
    print(f"Failed to import one or more modules: {e}")
try:
    from src.COND_PROB.src import private as runCondProb
    tasks["cond_prob"] = runCondProb
except (ImportError, ModuleNotFoundError) as e:
    print(f"Failed to import one or more modules: {e}")
try:
    from src.KMEANS import run as runKMeans
    tasks["kmeans"] = runKMeans
except (ImportError, ModuleNotFoundError) as e:
    print(f"Failed to import one or more modules: {e}")
try:
    from src.LR_PVAL import run as runLRPval
    tasks["lr_pval"] = runLRPval
except (ImportError, ModuleNotFoundError) as e:
    print(f"Failed to import one or more modules: {e}")



def main(targets):
    with open(os.path.join("config", "run.json")) as fh:
        params = json.load(fh)

    all_methods = ["lasso", "cond_prob", "kmeans", "lr_pval"]
    all_objects = [runLasso, runCondProb, runKMeans, runLRPval]

    # if no target is specified, run all methods
    target_methods = set(all_methods).intersection(set(targets))
    if len(set(all_methods).intersection(set(targets))) == 0:
        if len(set(target_methods).intersection(set(tasks.keys()))) == 0:
            print("Tasks not imported")
        task_config = params.get("task")
        if not task_config is None or len(task_config) == 0:
            target_methods = task_config
        else:
            print("No tasks specified in json or terminal")
        print("Run all data tasks:")

    combined_df = []
    for method, obj in tasks.items():
        if method not in target_methods:
            continue

        if params.get("verbose"):
            print(f"Running {method}")
        combine_df = obj.main(**params)
        combined_df.append(combine_df)

    fp = params.get("output")
    try:
        return combine_plot(combined_df, fp)
    except Exception as e:
        print(f"Error plotting: {e}")



def combine_plot(combined_df, fp):
    big_df = pd.concat(combined_df)
    plt.clf()
    plot = sns.lineplot(
        x="epsilon",
        y="utility",
        hue="task",
        style="task",
        data=big_df,
        markers=True,
    )
    plt.xscale("log")
    plt.ylim(0, 1)
    plt.title("Utility vs Epsilon")
    plt.xlabel("Epsilon (log scale)")
    plt.ylabel("Normalized Utility")

    plt.tight_layout()
    plt.savefig(os.path.join(*fp, "results.png"))
    return big_df


if __name__ == "__main__":
    targets = sys.argv[1:]
    # Targets: "lasso", "cond_prob", "kmeans", "lr_pval"
    main(targets)
