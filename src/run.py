#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import sys
import gym
import json
import importlib
import numpy as np

import utils
import monitor

ENVS = {
    "blackjack": "Blackjack-v1",
    "cliffwalking": "CliffWalking-v0",
    "taxi": "Taxi-v3",
}

def get_cmd_line():
    cmd_line = "python"

    for arg in sys.argv:
        cmd_line += " " + arg

    return cmd_line

def run_human_env(env, agent):
    runEnv = utils.RunEnv(env, agent)
    runEnv.play()

    return None

def main():
    # Get arguments
    args = utils.parse_arguments()

    # Create gym environment:
    env = gym.make(ENVS[args.environment])

    # Create new agent:
    agent_lib = importlib.import_module("agent." + args.agent)
    agent_class = agent_lib.Agent

    # Run the environment:
    if args.agent == "human":
        results = run_human_env(env, agent_class(env.action_space.n))
        return
    else:
        bench = monitor.Bench(env, agent_class, nb_bench=args.average, nb_episodes=args.episodes, window=1000)
        results = bench.run_bench()

    # Add cmd line to the result:
    results["cmd"] = get_cmd_line()
    results["agent"] = args.agent
    results["environment"] = args.environment

    # Save results
    if args.filename and results:
        with open(args.filename, "w") as outfile:
            json.dump(results, outfile)

if __name__ == "__main__":
    # execute only if run as a script
    main()