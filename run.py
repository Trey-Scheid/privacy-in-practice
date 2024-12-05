#!/usr/bin/env python

import sys
import json

from GD import GD
from SGD import SGD
from FTRLM import FTRLM
from ObjectivePerturbation import ObjPert


def main(targets):
    with open("params.json") as fh:
        data_params = json.load(fh)

    all_methods = ["gd", "sgd", "ftrlm", "objpert", "outpert"]

    # if no target is specified, run all methods
    target_methods = set(all_methods).intersection(set(targets))
    if len(set(all_methods).intersection(set(targets))) == 0:
        target_methods = all_methods

    if "gd" in target_methods:
        gd = GD(fp=data_params["fp"])


if __name__ == "__main__":
    targets = sys.argv[1:]
    # Targets: gd, sgd, ftrlm, objpert, outpert
    main(targets)
