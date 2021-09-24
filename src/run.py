#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

from cmd import parse_arguments
from monitor import Monitor
from play import RunEnv

import gym
import importlib
import numpy as np

def main():
    # Get arguments
    args = parse_arguments()

    # Create gym environment:
    env = gym.make('Blackjack-v1')

    # Create new agent
    agent_lib = importlib.import_module("agent." + args.agent)
    agent = agent_lib.Agent(env.action_space.n)

    # Run the environment:
    if args.agent == "human":
        runEnv = RunEnv(env, agent)
        runEnv.play()
    else:
        monitor = Monitor(env, agent, nb_episodes=args.episodes)
        avg_rewards, best_avg_reward = monitor.interact()

if __name__ == "__main__":
    # execute only if run as a script
    main()