#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import numpy as np
import matplotlib.pyplot as plt

def plot_result(results_list):
    plt.title(' vs '.join([res['agent'] for res in results_list]), loc='center')

    for results in results_list:
        plt.plot(np.arange(len(results['average_rewards'])), results['average_rewards'], label = results['agent'])

    plt.xlabel('Episodes')
    plt.ylabel('Rewards')

    plt.legend()
    plt.show()
