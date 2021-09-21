#!/usr/bin/python3
# ‑∗‑ coding: utf‑8 ‑∗‑

import sys
import numpy as np

from collections import deque

class RunEnv:

    def __init__(self, env, agent):
        """ Initialize the monitoring class.
        Params
        ======
        - env: instance of OpenAI Gym's Taxi-v3 environment
        - agent: agent that will interact with the environment.
        """
        self.env = env
        self.agent = agent

    def play(self):
        # begin the episode
        state = self.env.reset()
        reward = 0

        while True:
            print("State:", state, "|| Reward:", reward)
            next_state, reward, done, _ = self.play_step(state)

            # update the state (s <- s') to next time step
            state = next_state

            if done:
                break
        
        print("State:", state, "|| Reward:", reward)

    def play_step(self, state):
        # agent selects an action
        action = self.agent.select_action(state)
        # agent performs the selected action
        next_state, reward, done, _ = self.env.step(action)
        # agent performs internal updates based on sampled experience
        self.agent.step(state, action, reward, next_state, done)

        return next_state, reward, done, _
