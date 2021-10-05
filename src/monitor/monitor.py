#!/usr/bin/python3
# ‑∗‑ coding: utf‑8 ‑∗‑

import sys
import numpy as np

from collections import deque

class Monitor:
    """ Initialize the monitoring class.

    Args:
        env: instance of OpenAI Gym's environment
        agent: agent that will interact with the environment.
        nb_episodes: number of episodes of agent-environment interaction
        window: number of episodes to consider when calculating average rewards.

    Attributes:
        env: instance of OpenAI Gym's environment
        agent: agent that will interact with the environment.
        nb_episodes: number of episodes of agent-environment interaction
        window: number of episodes to consider when calculating average rewards.
    """
    def __init__(self, env, agent, nb_episodes=20000, window=100):
        self.env = env
        self.agent = agent
        self.nb_episodes = nb_episodes
        self.window = window

        self.avg_rewards = deque(maxlen=nb_episodes)
        self.sample_rewards = deque(maxlen=window)

        self.results = {
            "best_average_reward": np.NINF,
            "average_rewards": [],
            "epsilon": [],
        }

    def print_progress(self, i_episode):
        """ Monitor progress.
        """
        print("\rEpisode: {}/{} || Best average reward: {} || Average reward: {}"
            .format(i_episode + 1, self.nb_episodes, self.results["best_average_reward"], self.results["average_rewards"][-1]), end="")
        sys.stdout.flush()

    def step(self, state):
        action = self.agent.select_action(state)

        # agent performs the selected action
        next_state, reward, done, info = self.env.step(action)

        # agent performs internal updates based on sampled experience
        self.agent.step(state, action, reward, next_state, done)

        return next_state, reward, done, info

    def episode(self):
        # begin the episode
        state = self.env.reset()
        # initialize the sampled reward
        samp_reward = 0

        while True:
            next_state, reward, done, _ = self.step(state)
            # update the sampled reward
            samp_reward += reward
            # update the state (s <- s') to next time step
            state = next_state

            if done:
                # save final sampled reward
                self.sample_rewards.append(samp_reward)
                break

    def interact(self):
        """ Monitor agent's performance.
        Returns
            results: information of the training score and average results
        """
        for i_episode in range(0, self.nb_episodes):
            self.episode()

            if (i_episode % self.window) == 0:
                # get average reward from last 100 episodes
                avg_reward = np.mean(self.sample_rewards)
                self.results["average_rewards"].append(avg_reward)

                # update best average reward
                if avg_reward > self.results["best_average_reward"]:
                    self.results["best_average_reward"] = avg_reward

            self.print_progress(i_episode)

        print('\n')

        self.results["average_rewards"] = np.array(self.results["average_rewards"])[1:]

        return self.results
