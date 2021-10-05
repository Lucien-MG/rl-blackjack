#!/usr/bin/python3
# ‑∗‑ coding: utf‑8 ‑∗‑

import sys
import monitor
import numpy as np

from collections import deque

class Bench:
    """ Initialize the bench class.

    Args:
        env: instance of OpenAI Gym's environment
        agent: agent that will interact with the environment.
        nb_bench: number of episodes of agent-environment interaction
        nb_episodes: number of episodes of agent-environment interaction
        window: number of episodes to consider when calculating average rewards.

    Attributes:
        env: instance of OpenAI Gym's environment
        agent: agent that will interact with the environment.
        nb_episodes: number of episodes of agent-environment interaction
        window: number of episodes to consider when calculating average rewards.
    """
    def __init__(self, env, agent_class, agent_parameters=None, nb_bench=10, nb_episodes=20000, window=100):
        self.env = env
        self.agent_class = agent_class
        self.agent_parameters = agent_parameters
        self.nb_bench = nb_bench
        self.nb_episodes = nb_episodes
        self.window = window

        self.results = {
            "best_average_reward": np.NINF,
            "average_rewards": [],
        }

    def init_agent(self):
        if self.agent_parameters:
            return self.agent_class(self.env.action_space.n, **self.agent_parameters)
        else:
            return self.agent_class(self.env.action_space.n)

    def run_bench(self):
        results_list = []

        print("Bench: {}/{}".format(1, self.nb_bench))

        agent = self.init_agent()
        monitor_env = monitor.Monitor(self.env, agent, nb_episodes=self.nb_episodes)

        results = monitor_env.interact()

        for i_bench in range(2, self.nb_bench+1):
            print("Bench: {}/{}".format(i_bench, self.nb_bench))

            agent = self.init_agent()
            monitor_env = monitor.Monitor(self.env, agent, nb_episodes=self.nb_episodes, window=self.window)

            tmp_results = monitor_env.interact()

            results["best_average_reward"] += (1/i_bench) * (tmp_results["best_average_reward"] - results["best_average_reward"])
            results["average_rewards"] += (1/i_bench) * (tmp_results["average_rewards"] - results["average_rewards"])

        results["average_rewards"] = np.round(results["average_rewards"], 3)
        results["average_rewards"] = list(results["average_rewards"])

        return results