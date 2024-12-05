#!/usr/bin/env python

import sys
import json

from GD import GD
from SGD import SGD
from FTRLM import FTRLM
from ObjectivePerturbation import ObjPert


def main(targets):
    with open("params.json") as fh:
        params = json.load(fh)

    all_methods = ["gd", "sgd", "ftrlm", "objpert", "outpert"]
    all_objects = [GD, SGD, FTRLM, ObjPert]

    # if no target is specified, run all methods
    target_methods = set(all_methods).intersection(set(targets))
    if len(set(all_methods).intersection(set(targets))) == 0:
        target_methods = all_methods

    combined_df = []
    for method, obj in zip(all_methods, all_objects):
        if method not in target_methods:
            continue
        specific_params = {
            "fp": params["fp"],
            "epsilons": params["epsilons"],
            "delta": params["delta"],
        }
        if method == "objpert":
            specific_params["trials"] = params["ObjPert_trials"]
            specific_params["repeats"] = params["ObjPert_repeats"]

        run = obj(**specific_params)
        combine_df = run.run_all_plots()
        combined_df.append(combine_df)


if __name__ == "__main__":
    targets = sys.argv[1:]
    # Targets: gd, sgd, ftrlm, objpert, outpert
    main(targets)
