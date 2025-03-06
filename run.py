#!/usr/bin/env python

import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.LASSO import run as runLasso
from src.COND_PROB.src import private as runCondProb
from src.KMEANS import run as runKMeans
from src.LR_PVAL import run as runLRPval


def main(targets):
    with open("config.json") as fh:
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
    combine_plot(combined_df, fp)


def combine_plot(combined_df, fp):
    big_df = pd.concat(combined_df)
    plt.clf()
    plot = sns.lineplot(
        x="Epsilon", y="Error", hue="Method", style="Method", data=big_df, markers=True
    )
    plt.xscale("log")
    plt.ylim(0, 1)
    plt.title("Utility vs Epsilon")
    plt.xlabel("Epsilo (log scale)")
    plt.ylabel("Normalized Utility")

    plt.tight_layout()
    plt.savefig(fp + "results.png")


if __name__ == "__main__":
    targets = sys.argv[1:]
    # Targets: gd, sgd, ftrlm, objpert, outpert
    main(targets)
