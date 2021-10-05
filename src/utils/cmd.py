#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the environement with the chosen agent.')

    parser.add_argument('environment', metavar='environment', type=str,
                    help='Choose the environment to solve')

    parser.add_argument('agent', metavar='agent', type=str,
                    help='Choose agent to solve the environment')

    parser.add_argument('-n', metavar='nb_episodes', dest="episodes", type=int,
                    help='Set the number of episodes')

    parser.add_argument('-s', '--save', metavar='filename', dest="filename", type=str,
                    help='Save the results of the instance')

    parser.add_argument('-a', '--average', metavar='nb_episodes', dest="average", nargs='?', const=1, type=int, default=1,
                    help='Smoothed the results with "nb_episodes"')

    args = parser.parse_args()
    return args