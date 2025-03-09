#!/usr/bin/env python

import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.LASSO import run as runLasso
from src.COND_PROB.src import private as runCondProb
from src.KMEANS import run as runKMeans
from src.LR_PVAL import run as runLRPval



def main(targets):
    with open(os.path.join("config", "run.json")) as fh:
        params = json.load(fh)

    all_methods = ["lasso", "cond_prob", "kmeans", "lr_pval"]
    all_objects = [runLasso, runCondProb, runKMeans, runLRPval]

    # if no target is specified, run all methods
    target_methods = set(all_methods).intersection(set(targets))
    if len(set(all_methods).intersection(set(targets))) == 0:
        target_methods = all_methods

    combined_df = []
    for method, obj in zip(all_methods, all_objects):
        if method not in target_methods:
            continue

        print(f"Running {method}")
        combine_df = obj.main(**params)
        combined_df.append(combine_df)

    fp = params.get("output")
    try:
        combine_plot(combined_df, fp)
    except Exception as e:
        print(f"Error plotting {method}: {e}")



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
    plt.savefig(os.path.join(fp[0],fp[1], "results.png"))


if __name__ == "__main__":
    targets = sys.argv[1:]
    # Targets: "lasso", "cond_prob", "kmeans", "lr_pval"
    main(targets)
