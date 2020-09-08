import argparse
import numpy as np
import joblib
import copy
import time
import os
import gym

import dexterous_gym #load envs (install with pip install dexterous-gym)

from SCDM.TOPDM.envs.multiprocess_env import SubprocVecEnv
from SCDM.TOPDM.envs.util import CloudpickleWrapper
from SCDM.TOPDM.planner import Planner

class TrajectoryGenerator(object):
    def __init__(self, planner, tau=20, anneal_its=False, anneal_end=30, anneal_frac=0.5, num_searches=1,
                 num_samples_per_search=1000, num_iterations=40, sync_searches_every=10, one_search_after=20,
                 beta=0.7, tau_scaler=1.0, success_reward=True, min_noise=0.3, max_noise=0.3, min_noise_frac=0.3,
                 max_noise_frac=0.3, frac_best=0.05, traj_len=75, expt_name="", dir_name=""):
        self.planner = planner
        self.primary_env = self.planner.envs.dummy
        self.tau = tau
        self.anneal_its = anneal_its
        self.anneal_end = anneal_end
        self.anneal_frac = anneal_frac
        self.num_searches = num_searches
        self.num_samples_per_search = num_samples_per_search
        self.num_iterations = num_iterations
        self.sync_searches_every = sync_searches_every
        self.one_search_after = one_search_after
        self.beta = beta
        self.tau_scaler = tau_scaler
        self.success_reward = success_reward
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.min_noise_frac = min_noise_frac
        self.max_noise_frac = max_noise_frac
        self.frac_best = frac_best
        self.traj_len = traj_len
        self.expt_name = expt_name
        self.dir_name = dir_name
        os.makedirs("experiments/"+self.dir_name, exist_ok=True)
        self.log = open("experiments/"+self.dir_name+"/"+self.expt_name+"_log.txt", "a+")

    def generate_trajectory(self, goal=None, start_state=None, verbose=True):
        if goal is None:
            goal = self.primary_env.env._sample_goal()
        if start_state is None:
            self.primary_env.reset()
            start_states = [CloudpickleWrapper(self.primary_env.env.sim.get_state()) for _ in range(self.planner.num_envs)]
        else:
            if isinstance(start_state, CloudpickleWrapper):
                start_states = [copy.deepcopy(start_state) for _ in range(self.planner.num_envs)]
            elif not isinstance(start_state, list):
                start_states = [CloudpickleWrapper(copy.deepcopy(start_state)) for _ in range(self.planner.num_envs)]
        self.goal = goal
        self.planner.envs.set_goals(goal)
        self.primary_env.env.goal = goal
        self.primary_env.goal = goal
        self.trajectory = {
            "sim_states": [start_states[0]], "actions": [], "rewards": [], "success": [], "goal": goal,
            "best_rewards_it": []
        }
        curr_states = copy.deepcopy(start_states)

        if self.anneal_its:
            total_its = self.num_iterations * self.traj_len
            anneal_its = int(self.anneal_frac*total_its) #separate iterations above min baseline
            min_its = (total_its - anneal_its) // self.traj_len
            y0 = 2 * anneal_its / self.anneal_end
        prev_actions = None
        prev_best_ac_mean = None

        for t in range(self.traj_len):
            t1 = time.time()
            if self.anneal_its:
                num_its = int(min_its + (y0-(y0/self.anneal_end)*t))
                num_its = max(min_its, num_its)
            else:
                num_its = self.num_iterations
            best_rewards, best_ac_trajs, best_ac_means, best_rewards_it = self.planner.plan(
                curr_states, prev_actions, prev_best_ac_mean, tau=self.tau, num_searches=self.num_searches,
                num_samples_per_search=self.num_samples_per_search, num_iterations=num_its,
                sync_searches_every=self.sync_searches_every, one_search_after=self.one_search_after,
                beta=self.beta, tau_scaler=self.tau_scaler, success_reward=self.success_reward,
                min_noise=self.min_noise, max_noise=self.max_noise, min_noise_frac=self.min_noise_frac,
                max_noise_frac=self.max_noise_frac, frac_best=self.frac_best
            )
            self.trajectory["best_rewards_it"].append(best_rewards_it.copy())
            action = best_ac_trajs[0][0, :].copy()

            prev_actions = np.tile(action, (self.num_searches,1))
            prev_best_ac_mean = best_ac_means[0][1:, ...]

            curr_states, reward, success, (d_pos, d_rot) = self.step_trajectory(curr_states, action)
            joblib.dump((self.trajectory, prev_best_ac_mean), "experiments/"+self.dir_name+"/"+self.expt_name+".pkl")
            t2 = time.time()

            if verbose:
                write_out = "Step %d. Reward: %f, success: %r, d_pos: %f, d_rot: %f. Best future reward sum: %f. Took %f seconds\n" % \
                            (t, reward, success, d_pos, d_rot, best_rewards[0], t2-t1)
                self.log.write(write_out)
                print(write_out)
        return self.trajectory

    def step_trajectory(self, curr_states, action):
        self.primary_env.env.sim.set_state(curr_states[0].x)
        self.primary_env.env.sim.forward()
        obs, r, _, info = self.primary_env.step(action)
        try:
            d_pos, d_rot = self.primary_env.env._goal_distance(obs["achieved_goal"], self.primary_env.goal)
        except:
            d_pos, d_rot = 0.0, 0.0
        if self.planner.name is not None:
            if self.planner.name.startswith("Two"):
                d_pos = [d_pos[0], d_rot[0]];
                d_rot = [d_pos[1],
                         d_rot[1]]  # goal_distance returns (dpos1, drot1), (dpos2, drot2) when there are two objects
                d_pos[0] *= 20;
                d_pos[1] *= 20
                d_pos = np.mean(d_pos);
                d_rot = np.mean(d_rot)
            else:
                d_pos *= 20
        else:
            d_pos *= 20.0
        state = [CloudpickleWrapper(self.primary_env.env.sim.get_state()) for _ in range(self.planner.num_envs)]
        self.trajectory["sim_states"].append(state)
        self.trajectory["actions"].append(action)
        self.trajectory["rewards"].append(r)
        try:
            self.trajectory["success"].append(info["is_success"])
        except:
            self.trajectory["success"].append(None) #PenSpin doesn't have success criteria
        return state, r, self.trajectory["success"][-1], (d_pos, d_rot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_name', type=str, default="tests", help="experiment saved in experiments/dir_name/")
    parser.add_argument('--expt_name', type=str, default="default", help="results saved as experiments/dir_name/expt_name.pkl")
    parser.add_argument('--env', type=str, default="EggHandOver-v0", help="environment identifier")
    parser.add_argument('--tau', type=int, default=20, help="number of future steps to model for each planning step")
    parser.add_argument('--num_searches', type=int, default=1, help="number of parallel searches at each planning step")
    parser.add_argument('--num_samples_per_search', type=int, default=1000, help="number of trajectories per iteration per planning step")
    parser.add_argument('--sync_searches_every', type=int, default=10, help="sync parallel searches every this many its")
    parser.add_argument('--one_search_after', type=int, default=20, help="after this many iterations, continue with only one search")
    parser.add_argument('--base_iterations', type=int, default=40, help="base number of iterations per planning step")
    parser.add_argument('--frac_best', type=float, default=0.05, help="fraction of best trajectories that are kept")
    parser.add_argument('--init_noise', type=float, default=0.9, help="initial random noise when starting a trajectory")
    parser.add_argument('--beta', type=float, default=0.7, help="coupling coefficient between actions at different steps")
    parser.add_argument('--min_noise', type=float, default=0.3, help="minimum noise added at each iteration")
    parser.add_argument('--max_noise', type=float, default=0.3, help="maximum noise added at each iteration")
    parser.add_argument('--min_noise_frac', type=float, default=0.3, help="minimum fraction of action traj that has noise added to it")
    parser.add_argument('--max_noise_frac', type=float, default=0.3, help="maximum fraction of action traj that has noise added to it")
    parser.add_argument('--num_envs', type=int, default=1, help="number of parallel env instances when modelling future trajectories")
    parser.add_argument('--traj_len', type=int, default=50, help="trajectory length")
    parser.add_argument('--tau_scaler', type=float, default=1.0, help="scale future rewards (can be positive/negative)")
    parser.add_argument('--anneal_end', type=int, default=-1, help="if annealing number of iterations, point when we stop. -1 is halfway")
    parser.add_argument('--anneal_frac', type=int, default=0.5, help="if annealing iterations, what fraction of total is used as extra")
    parser.add_argument('--linear_anneal_its', dest='anneal_its', action='store_true', help="whether to anneal no. iterations as traj progresses")
    parser.add_argument('--no_success_reward', dest='success_reward', action='store_true', help="extra reward if goal is achieved")
    parser.set_defaults(anneal_its=False, success_reward=True)
    args = parser.parse_args()
    anneal_end = args.anneal_end if args.anneal_end > 0 else (args.traj_len // 2)

    def env_fn():
        return gym.make(args.env)

    env_fns = [env_fn for _ in range(args.num_envs)]
    print(args.expt_name)
    print("\n")
    envs = SubprocVecEnv(env_fns)

    planner = Planner(envs, name=args.env, init_noise=args.init_noise)
    generator = TrajectoryGenerator(planner, tau=args.tau, anneal_its=args.anneal_its, anneal_end=args.anneal_end,
                                    anneal_frac=args.anneal_frac, num_searches=args.num_searches,
                                    num_samples_per_search=args.num_samples_per_search, num_iterations=args.base_iterations,
                                    sync_searches_every=args.sync_searches_every, one_search_after=args.one_search_after,
                                    beta=args.beta, tau_scaler=args.tau_scaler, success_reward=args.success_reward,
                                    min_noise=args.min_noise, max_noise=args.max_noise, min_noise_frac=args.min_noise_frac,
                                    max_noise_frac=args.max_noise_frac, frac_best=args.frac_best, traj_len=args.traj_len,
                                    expt_name=args.expt_name, dir_name=args.dir_name)
    generator.generate_trajectory()