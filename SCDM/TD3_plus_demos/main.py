#built directly on top of author's TD3 implementation: https://github.com/sfujim/TD3

import numpy as np
import torch
import gym
import argparse
import os
import dexterous_gym
import joblib

import SCDM.TD3_plus_demos.utils as utils
import SCDM.TD3_plus_demos.TD3 as TD3


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state = eval_env.reset()
		num_steps = 0
		prev_action = np.zeros((eval_env.action_space.shape[0],))
		while num_steps < eval_env._max_episode_steps:
			action = policy.select_action(np.array(state), prev_action)
			state, reward, done, _ = eval_env.step(action)
			prev_action = action.copy()
			avg_reward += reward
			num_steps += 1

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="PenSpin-v0")              # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25000, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5000, type=int)      # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=10e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.98)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

	#my parameters
	parser.add_argument("--beta", type=float, default=0.7)          # action coupling parameter
	parser.add_argument("--pd_init", type=float, default=0.7)       # initial probability of loading to demo for each segment
	parser.add_argument("--pr_init", type=float, default=0.2)       # initial probability of sampling segment from reset
	parser.add_argument("--pd_decay", type=float, default=0.999996)     # after each segment scale probability down by this amount
	parser.add_argument("--pr_decay", type=float, default=0.999996)     # after each segment scale probability down by this amount
	parser.add_argument("--segment_len", type=int, default=15)      # how long each "segment" is ran before resampling
	parser.add_argument("--use_normaliser", dest='use_normaliser', action='store_true') #use demos to normalise observations
	parser.add_argument("--update_normaliser_every", type=int, default=20)
	parser.add_argument("--expt_tag", type=str, default="")
	parser.set_defaults(use_normaliser=False)
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}_{args.expt_tag}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env_main = gym.make(args.env)
	env_demo = gym.make(args.env)
	env_reset = gym.make(args.env)

	demo_states = []
	demo_prev_actions = []
	files = os.listdir("demonstrations")
	files = [file for file in files if file.endswith(".pkl")]
	for file in files:
		traj = joblib.load("demonstrations/" + file)
		for k, state in enumerate(traj["sim_states"]):
			demo_states.append(state)
			if k==0:
				prev_action = np.zeros((env_main.action_space.shape[0],))
			else:
				prev_action = traj["actions"][k-1]
			demo_prev_actions.append(prev_action.copy())

	# Set seeds
	env_main.seed(args.seed)
	env_demo.seed(args.seed+1)
	env_reset.seed(args.seed+2)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env_main.observation_space.shape[0]
	action_dim = env_main.action_space.shape[0]
	max_action = float(env_main.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		"beta": args.beta
	}

	# Initialize policy
	# Target policy smoothing is scaled wrt the action scale
	policy = TD3.TD3(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]

	total_timesteps = 0
	segment_timestep = 0
	main_episode_timesteps = 0
	main_episode_prev_ac = np.zeros((action_dim,))
	main_episode_obs = env_main.reset()

	pd_prob = args.pd_init
	pr_prob = args.pr_init

	for t in range(int(args.max_timesteps)):
		
		if segment_timestep % args.segment_len == 0:
			segment_timestep = 0
			if t > 0:
				pd_prob *= args.pd_decay
				pr_prob *= args.pr_decay
			rn = np.random.rand()
			if rn < pd_prob:
				segment_type = "pd"
			elif rn < pd_prob + pr_prob:
				segment_type = "pr"
			else:
				segment_type = "full"
			if segment_type == "pd":
				ind = np.random.randint(0, len(demo_states))
				env_demo.reset()
				env_demo.env.sim.set_state(demo_states[ind])
				env_demo.env.sim.forward()
				observation = env_demo.env._get_obs()["observation"]
				prev_action = demo_prev_actions[ind]
			elif segment_type == "pr":
				observation = env_reset.reset()
				prev_action = np.zeros((action_dim,))
			else:
				prev_action = main_episode_prev_ac
				observation = main_episode_obs

		if t < args.start_timesteps:
			action = (
				policy.beta*env_main.action_space.sample() + (1-policy.beta)*prev_action
			).clip(-max_action, max_action)
		else:
			noise = np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			action = (
				policy.select_action(observation, prev_action, noise=noise)
			).clip(-max_action, max_action)

		segment_timestep += 1
		total_timesteps += 1

		if segment_type == "pd":
			next_observation, reward, _, _ = env_demo.step(action)
		elif segment_type == "pr":
			next_observation, reward, _, _ = env_reset.step(action)
		else:
			main_episode_timesteps += 1
			next_observation, reward, _, _ = env_main.step(action)
			main_episode_prev_ac = action.copy()
			episode_obs = next_observation.copy()
		replay_buffer.add(observation, action, next_observation, reward, prev_action)
		if args.use_normaliser:
			policy.normaliser.update(observation)
			if t % args.update_normaliser_every == 0:
				policy.normaliser.recompute_stats()
		prev_action = action.copy()
		observation = next_observation.copy()
		if segment_type == "full":
			if main_episode_timesteps == env_main._max_episode_steps:
				main_episode_timesteps = 0
				prev_action = main_episode_prev_ac = np.zeros((action_dim,))
				observation = episode_obs = env_main.reset()

		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if (t+1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")
			print("Evaluation after %d steps - average reward: %f" % (total_timesteps, evaluations[-1]))