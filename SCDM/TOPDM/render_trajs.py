#render all files from directory of experiments (assume all of same environment)

import joblib
import gym
import argparse
import os
import time
import dexterous_gym

parser = argparse.ArgumentParser()
parser.add_argument('--dir_name', type=str, default="tests", help="experiment saved in experiments/dir_name/")
parser.add_argument('--expt_name', type=str, default="none", help="experiment id. if none show all in dir")
parser.add_argument('--env', type=str, default="EggHandOver-v0")
parser.add_argument('--delay', type=float, default=0.03, help="time between frames")
args = parser.parse_args()

if args.expt_name == "none":
    files = os.listdir("experiments/"+args.dir_name)
    files = [file for file in files if file.endswith(".pkl")]
    experiments = ["experiments/"+args.dir_name+"/"+file for file in files]
else:
    experiments = ["experiments/"+args.dir_name+"/"+args.expt_name]

env = gym.make(args.env)

for experiment in experiments:
    traj = joblib.load(experiment)[0]
    env.reset()
    env.env.sim.set_state(traj["sim_states"][0].x)
    if env == "PenSpin-v0":
        import numpy as np
        env.goal = np.array([10000, 10000, 10000, 0, 0, 0, 0])
        env.env.goal = np.array([10000, 10000, 10000, 0, 0, 0, 0])
    else:
        env.goal = traj["goal"]
        env.env.goal = traj["goal"]
    env.render()
    for action in traj["actions"]:
        env.step(action)
        time.sleep(args.delay)
        env.render()