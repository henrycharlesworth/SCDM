import SCDM.TD3_plus_demos.TD3 as TD3
import dexterous_gym
import gym
import numpy as np
import time

filename = "models/TD3_PenSpin-v0_0_beta_0_7_norm"
beta = 0.7
env_name = "PenSpin-v0"

env = gym.make("PenSpin-v0")
steps = 1000 #long run, "standard" episode is 250

def eval_policy(policy, env_name, seed, eval_episodes=1, render=True, delay=0.0):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state = eval_env.reset()
        if render:
            eval_env.render()
            time.sleep(delay)
        num_steps = 0
        prev_action = np.zeros((eval_env.action_space.shape[0],))
        while num_steps < steps:
            action = policy.select_action(np.array(state), prev_action)
            state, reward, done, _ = eval_env.step(action)
            if render:
                eval_env.render()
                time.sleep(delay)
            prev_action = action.copy()
            avg_reward += reward
            num_steps += 1
            print(num_steps)

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

kwargs = {
    "state_dim": env.observation_space.shape[0],
    "action_dim": env.action_space.shape[0],
    "beta": beta,
    "max_action": 1.0
}

policy = TD3.TD3(**kwargs)
policy.load(filename)

eval_policy(policy, env_name, seed=0, eval_episodes=1, render=True, delay=0.03)