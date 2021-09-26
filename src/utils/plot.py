#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import numpy as np
import matplotlib.pyplot as plt

def plot_result(results):
    plt.plot(np.arange(len(results['average_rewards'])), results['average_rewards'])
    plt.ylabel('some numbers')
    plt.show()
