import numpy as np
import copy
import time
import joblib
import gym

from SCDM.TOPDM.envs.util import CloudpickleWrapper
from SCDM.TOPDM.humanoid_expts.planner import Planner
from SCDM.TOPDM.humanoid_expts.envs.multiprocess_env import SubprocVecEnv

class TrajectoryGenerator(object):
    def __init__(self, planner, traj_len=50, expt_tag="default", goal_based=True, mask_done=False):
        self.planner = planner
        self.traj_len = traj_len
        self.num_envs = self.planner.envs.num_envs
        self.primary_env = self.planner.envs.dummy
        self.goal_based = goal_based
        self.mask_done = mask_done
        self.tau = copy.copy(self.planner.tau)
        self.log = open("results/"+planner.name+"_"+expt_tag+"_log.txt", "a+")

    def generate_trajectory(self, goal=None, start_state=None, verbose=True):
        self.planner.prev_actions = np.zeros((self.planner.ac_dim,))
        self.planner.tau = self.tau
        self.updated_tau = False
        self.use_best = False
        self.planner.prev_best_ac_mean = None
        self.primary_env.reset()
        if self.goal_based:
            if goal is None:
                goal = self.primary_env.env._sample_goal()
            self.goal = goal
        else:
            goal = None
            self.goal = None
        if start_state is None:
            self.primary_env.reset()
            start_states = [CloudpickleWrapper(self.primary_env.env.sim.get_state()) for _ in range(self.num_envs)]
        else:
            if isinstance(start_state, CloudpickleWrapper):
                start_states = [copy.deepcopy(start_state) for _ in range(self.num_envs)]
            elif not isinstance(start_state, list):
                start_states = [CloudpickleWrapper(copy.deepcopy(start_state)) for _ in range(self.num_envs)]
        if self.goal_based:
            self.planner.envs.set_goals(goal)
            self.primary_env.goal = goal
            self.primary_env.env.goal = goal
        self.trajectory = {
            "sim_states": [start_states[0]], "actions": [], "rewards": [], "success": [], "goal": goal,
            "best_rewards_it": []
        }
        curr_states = copy.deepcopy(start_states)

        sum_rewards = 0

        best_curr_reward = -np.inf
        best_curr_ac_seq = None

        # total_iterations = planner.num_iterations * self.traj_len
        # min_it = planner.num_iterations // 2
        # to_allocate = total_iterations - min_it*self.traj_len
        # max_step = self.traj_len // 2
        # y0 = int(2 * to_allocate / max_step)

        for t in range(self.traj_len):
            t1 = time.time()
            # self.planner.num_iterations = int(max(min_it + (y0 - (y0/max_step)*t), 1))
            # if self.planner.num_iterations < min_it:
            #     self.planner.num_iterations = min_it
            self.planner.later_noise = self.planner.later_noise[0]*np.ones((max(self.planner.num_iterations-1, 1),))
            self.planner.later_noise_frac = self.planner.later_noise_frac[0]*np.ones((max(self.planner.num_iterations-1,1),))

            best_reward, best_ac_traj, best_ac_means, best_reward_its, (mean_rew, std_rew) = self.planner.plan(curr_states)
            self.trajectory["best_rewards_it"].append(best_reward_its.copy())
            if self.use_best:
                if best_reward > best_curr_reward:
                    best_curr_reward = best_reward
                    best_curr_ac_seq = best_ac_traj[1:].copy()
                    action = best_ac_traj[0,:]
                else:
                    action = best_curr_ac_seq[0, :].copy()
                    if len(best_curr_ac_seq) == 1:
                        best_curr_reward = -np.inf
                        best_curr_ac_seq = None
                    else:
                        best_curr_ac_seq = best_curr_ac_seq[1:, ...]
            else:
                action = best_ac_traj[0, :]

            planner.prev_action = action.copy()
            planner.prev_best_ac_mean = best_ac_means[1:, ...]

            curr_states, reward, success, (d_pos, d_rot), d = self.step_trajectory(curr_states, action)

            sum_rewards += reward

            t2 = time.time()
            if verbose:
                write_out = "Step %d. Reward: %f, success: %r, d_pos: %f, d_rot: %f. Best future reward: %f. Return so far: %f. Done: %d. Took %f seconds!\n" % (t, reward, success, d_pos, d_rot, best_reward, sum_rewards, d, t2-t1)
                self.log.write(write_out)
                print(write_out)
            joblib.dump((self.trajectory, args), "results/" + args.env + "_" + args.expt_tag + ".pkl")

            if self.mask_done:
                if d:
                    return self.trajectory

        return self.trajectory

    def step_trajectory(self, curr_states, action):
        self.primary_env.env.sim.set_state(curr_states[0].x)
        self.primary_env.env.sim.forward()
        obs, r, d, info = self.primary_env.step(action)
        try:
            d_pos, d_rot = self.primary_env.env._goal_distance(obs["achieved_goal"], self.primary_env.goal)
        except:
            d_pos, d_rot = 0.0, 0.0 #PenSpin has no goal.
        if self.planner.name is not None:
            if self.planner.name.startswith("Two"):
                d_pos = [d_pos[0], d_rot[0]]; d_rot = [d_pos[1], d_rot[1]] #goal_distance returns (dpos1, drot1), (dpos2, drot2) whent here are two objects
                d_pos[0] *= 20; d_pos[1] *= 20
                d_pos = np.mean(d_pos); d_rot = np.mean(d_rot)
            else:
                d_pos *= 20
        else:
            d_pos *= 20.0
        state = [CloudpickleWrapper(self.primary_env.env.sim.get_state()) for _ in range(self.num_envs)]
        self.trajectory["sim_states"].append(state)
        self.trajectory["actions"].append(action)
        self.trajectory["rewards"].append(r)
        try:
            self.trajectory["success"].append(info["is_success"])
        except:
            self.trajectory["success"].append(None)  # PenSpin doesn't have success criteria
        return state, r, self.trajectory["success"][-1], (d_pos, d_rot), d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--expt_tag', type=str, default="default")
    parser.add_argument('--env', type=str, default="Humanoid-v3")
    parser.add_argument('--tau', type=int, default=20)
    parser.add_argument('--num_samples_per_it', type=int, default=1000)
    parser.add_argument('--frac_best', type=float, default=0.05)
    parser.add_argument('--num_iterations', type=int, default=40)
    parser.add_argument('--init_noise', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.7)
    parser.add_argument('--later_noise', type=float, default=0.3)
    parser.add_argument('--later_noise_frac', type=float, default=0.3)
    parser.add_argument('--forget_init_past_ac_frac', type=float, default=0.0)
    parser.add_argument('--no_sum_rewards', dest='sum_rewards', action='store_false')
    parser.add_argument('--initialise_frac_with_prev_best', type=float, default=1.0)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--traj_len', type=int, default=50)
    parser.add_argument('--tau_scaler', type=float, default=1.0)
    parser.add_argument('--not-goal-based', action='store_true', default=False)
    parser.add_argument('--mask-done', action='store_true', default=False)
    parser.set_defaults(sum_rewards=True)
    args = parser.parse_args()

    if args.not_goal_based:
        goal_based = False
    else:
        goal_based = True

    num_envs = args.num_envs

    print(args.env + " - " + args.expt_tag)
    print("\n")

    if args.env == "HandPenMod":
        from SCDM.TOPDM.envs.hand_pen import HandPen
        def env_fn():
            return HandPen()
    else:
        def env_fn():
            env = gym.make(args.env)
            env._max_episode_steps = np.inf
            return env
    env_fns = [env_fn for _ in range(args.num_envs)]
    envs = SubprocVecEnv(env_fns)

    planner = Planner(envs, tau=args.tau, num_samples_per_it=args.num_samples_per_it, frac_best=args.frac_best,
                      num_iterations=args.num_iterations, init_noise=args.init_noise,
                      beta=args.beta, later_noise=args.later_noise, later_noise_frac=args.later_noise_frac,
                      forget_init_past_ac_frac=args.forget_init_past_ac_frac, sum_rewards=args.sum_rewards,
                      initialise_frac_with_prev_best=args.initialise_frac_with_prev_best, name=args.env,
                      tau_scaler=args.tau_scaler, mask_done=args.mask_done)
    generator = TrajectoryGenerator(planner, traj_len=args.traj_len, expt_tag=args.expt_tag, goal_based=goal_based,
                                    mask_done=args.mask_done)
    traj = generator.generate_trajectory()
