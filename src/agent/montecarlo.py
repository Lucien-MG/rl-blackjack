#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import numpy as np
from collections import defaultdict, deque

class Agent:
    """ Initialize agent.
        
    Args:
        nb_actions (int): number of actions available to the agent
    """
    def __init__(self, nb_actions, epsilon=0.99, alpha=1e-2, gamma=1e-1, min_epsilon=1e-4, epsilon_decay_factor=1e-6):
        self.nb_actions = nb_actions
        self.q_table = defaultdict(lambda: np.zeros(self.nb_actions))
        self.memory = deque()

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_factor = epsilon_decay_factor

        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state) -> int:
        """ Given the state, select an action.

        Args:
            state (obj): the current state of the environment.
        
        Returns:
            action (int): an integer compatible with the task's action space.
        """
        if np.random.uniform() > self.epsilon:
            return np.argmax(self.q_table[state])

        return np.random.choice(self.nb_actions)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Args:
            state (obj): the previous state of the environment
            action (int): the agent's previous choice of action
            reward (float): last reward received
            next_state (obj): the current state of the environment
            done (bool): whether the episode is complete (True or False)
        """
        self.memory.appendleft((state, action, reward, next_state, done))

        if done:
            while len(self.memory):
                state, action, reward, next_state, done = self.memory.pop()

                target_reward = reward + self.gamma * np.mean(self.q_table[next_state])
                error = target_reward - self.q_table[state][action]

                self.q_table[state][action] += self.alpha * error
                self.epsilon = max(self.epsilon - self.epsilon_decay_factor, self.min_epsilon)
