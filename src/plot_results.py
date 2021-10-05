#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import sys
import gym
import json
import importlib
import numpy as np

import utils
import monitor

def main():
    results_list = []

    for i in range(1, len(sys.argv)):
        with open(sys.argv[i]) as f:
            results_list.append(json.load(f))

    # Plot results:
    utils.plot_result(results_list)

if __name__ == "__main__":
    # execute only if run as a script
    main()