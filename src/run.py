# from agent.human import Agent
from agent.montecarlo import Agent
from monitor import Monitor
from play import RunEnv

import gym
import numpy as np

def main():
    # Create gym environment:
    env = gym.make('Blackjack-v1')

    # Create new agent
    agent = Agent(env.action_space.n)

    # Create Monitor:
    monitor = Monitor(env, agent, nb_episodes=1000000)
    # runEnv = RunEnv(env, agent)

    # runEnv.play()

    # Run agent
    avg_rewards, best_avg_reward = monitor.interact()

if __name__ == "__main__":
    # execute only if run as a script
    main()