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

        self.avg_rewards = deque(maxlen=nb_episodes)
        self.best_avg_reward = np.NINF
        self.sample_rewards = deque(maxlen=window)

    def print_progress(self, i_episode):
        """ Monitor progress.
        """
        print("\rEpisode: {}/{} || Best average reward: {}".format(i_episode, self.nb_episodes, self.best_avg_reward), end="")
        sys.stdout.flush()

    def play_step(self, state):
        # agent selects an action
        action = self.agent.select_action(state)
        # agent performs the selected action
        next_state, reward, done, _ = self.env.step(action)
        # agent performs internal updates based on sampled experience
        self.agent.step(state, action, reward, next_state, done)

        return next_state, reward, done, _

    def interact(self):
        """ Monitor agent's performance.
        Returns
        - avg_rewards: deque containing average rewards
        - best_avg_reward: largest value in the avg_rewards deque
        """
        for i_episode in range(1, self.nb_episodes+1):
            # begin the episode
            state = self.env.reset()
            # initialize the sampled reward
            samp_reward = 0

            while True:
                next_state, reward, done, _ = self.play_step(state)

                # update the sampled reward
                samp_reward += reward

                # update the state (s <- s') to next time step
                state = next_state

                if done:
                    # save final sampled reward
                    self.sample_rewards.append(samp_reward)
                    break

            if (i_episode >= 100):
                # get average reward from last 100 episodes
                avg_reward = np.mean(self.sample_rewards)
                # append to deque
                self.avg_rewards.append(avg_reward)
                # update best average reward
                if avg_reward > self.best_avg_reward:
                    self.best_avg_reward = avg_reward

            self.print_progress(i_episode)

            if i_episode == self.nb_episodes:
                print('\n')
        
        return self.avg_rewards, self.best_avg_reward
