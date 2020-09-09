import joblib
import gym
import argparse
import os
import time
import dexterous_gym

delay = 0.03
env_name = "EggHandOver-v0"
env = gym.make(env_name)

file_names = [
    "EggHandOver-v0_traj_1.pkl",
    "EggHandOver-v0_traj_2.pkl"
]
trajectory_files = [
    "experiments/" + file for file in file_names
]

for traj in trajectory_files:
    traj = joblib.load(traj)[0]
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
        time.sleep(delay)
        env.render()