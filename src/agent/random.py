#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import numpy as np

class Agent:

    def __init__(self, nb_actions):
        """ Initialize agent.
        Params
        - nb_actions: number of actions available to the agent
        """
        self.nb_actions = nb_actions

    def select_action(self, state) -> int:
        """ Given the state, select an action.
        Params
        ======
        - state: the current state of the environment
        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(self.nb_actions)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        return
