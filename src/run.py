#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import gym
import json
import importlib
import numpy as np

import utils
import monitor

def run_human_env(env, agent, args):
    runEnv = utils.RunEnv(env, agent)
    runEnv.play()

def run_training_env(env, agent, args):
    monitor_env = monitor.Monitor(env, agent, nb_episodes=args.episodes)
    results = monitor_env.interact()

    if args.filename:
        with open(args.filename, "w") as outfile:
            json.dump(results, outfile)
    
    utils.plot_result(results)

def main():
    # Get arguments
    args = utils.parse_arguments()

    # Create gym environment:
    env = gym.make('Blackjack-v1')

    # Create new agent
    agent_lib = importlib.import_module("agent." + args.agent)
    agent = agent_lib.Agent(env.action_space.n)

    # Run the environment:
    if args.agent == "human":
        run_human_env(env, agent, args)
    else:
        run_training_env(env, agent, args)

if __name__ == "__main__":
    # execute only if run as a script
    main()