import gym
import time
import joblib

filename = "results/Humanoid-v3_results.pkl"
traj, _ = joblib.load(filename)

env = gym.make('Humanoid-v3')
env.reset()
env.env.sim.set_state(traj["sim_states"][0].x)
env.env.sim.forward()
env.render()

for z, action in enumerate(traj["actions"]):
    if z > 0:


        env.env.sim.set_state(traj["sim_states"][z][0].x)
    else:
        env.env.sim.set_state(traj["sim_states"][z].x)
    env.env.sim.forward()
    o, r, d, i = env.step(action)
    print("step {}, reward {}, done {}".format(z, r, d))
    time.sleep(0.02)
    env.render()
