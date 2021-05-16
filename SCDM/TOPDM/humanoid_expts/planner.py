import numpy as np

class Planner(object):
    def __init__(self, envs, tau, num_samples_per_it=2000, frac_best=0.05,
                 num_iterations=10, init_noise=0.9, beta=0.7, later_noise=0.1, later_noise_frac=0.3,
                 forget_init_past_ac_frac=0.5, sum_rewards=True, initialise_frac_with_prev_best=1.0,
                 name=None, tau_scaler=1.0, add_reverse_tau_scaler=False, mask_done=False):
        assert np.abs(int(num_samples_per_it*frac_best) - num_samples_per_it*frac_best) < 1e-10
        assert num_samples_per_it % envs.num_envs == 0
        self.envs = envs
        self.num_envs = envs.num_envs
        self.tau = tau
        self.ac_dim = self.envs.ac_dim
        self.num_samples_per_it = num_samples_per_it
        self.frac_best = frac_best
        self.num_iterations = num_iterations
        self.init_noise = init_noise
        self.forget_init_past_ac_frac = forget_init_past_ac_frac
        self.beta = beta
        self.sum_rewards = sum_rewards
        self.initialise_frac_with_prev_best = initialise_frac_with_prev_best
        self.prev_best_ac_mean = None
        self.reward_scaler = (tau_scaler)**(np.arange(tau)+1)
        self.mask_done = mask_done
        if add_reverse_tau_scaler:
            self.reward_scaler += np.flip(self.reward_scaler)
        self.name = name
        if isinstance(later_noise, float):
            self.later_noise = later_noise*np.ones((num_iterations-1,))
        else:
            assert len(later_noise) == num_iterations-1
            self.later_noise = later_noise
        if isinstance(later_noise_frac, float):
            self.later_noise_frac = later_noise_frac*np.ones((num_iterations-1,))
        else:
            assert len(later_noise_frac) == num_iterations-1
            self.later_noise_frac = later_noise_frac
        self.prev_action = np.zeros((self.ac_dim,))

    def plan(self, env_states):
        best_ac_traj = None
        best_ac_means = None
        best_reward = -np.inf
        curr_action_means = np.clip(self.init_noise*np.random.randn(self.num_samples_per_it, self.tau, self.ac_dim), -1.0, 1.0)
        if self.prev_best_ac_mean is not None:
            num_prev = int(self.initialise_frac_with_prev_best*curr_action_means.shape[0])
            curr_action_means[:num_prev, :self.tau-1, :] = self.prev_best_ac_mean[np.newaxis, :self.tau-1, :]
            shape = curr_action_means[:num_prev, :self.tau-1, :].shape
            curr_action_means[:num_prev, :self.tau-1, :] += (np.random.rand(*shape) < self.later_noise_frac[0]).astype(int)*self.later_noise[0]*np.random.randn(*shape)
            curr_action_means[:num_prev, :self.tau-1, :] = np.clip(curr_action_means[:num_prev, :self.tau-1, :], -1.0, 1.0)
        else:
            num_prev = 0

        num_other = self.num_samples_per_it - num_prev
        num_prev_best = int(num_prev*self.frac_best)
        num_best = int(self.frac_best * self.num_samples_per_it)
        num_other_best = num_best - num_prev_best

        num_copies = self.num_samples_per_it // num_best

        past_actions = np.tile(self.prev_action, (self.num_samples_per_it,1))
        forget_inds = np.where(np.random.rand(self.num_samples_per_it) < self.forget_init_past_ac_frac)[0]
        past_actions[forget_inds, :] = 0.0 #allow for new actions to be tried without constraining yourself to be following the previous action

        best_rewards_it = []

        for it in range(self.num_iterations):
            curr_rewards = np.zeros((self.num_samples_per_it,))
            curr_states = np.zeros((self.num_samples_per_it, self.tau, self.envs.state_dim))
            forget_inds = np.where(np.sum(np.abs(past_actions), axis=-1) < 1e-8)[0]
            all_actions = np.zeros((self.num_samples_per_it, self.tau, self.ac_dim))
            for i in range(self.tau):
                if i==0:
                    all_actions[:, i, :] = self.beta*(curr_action_means[:, i, :]) + (1-self.beta)*past_actions
                    all_actions[forget_inds, i, :] = curr_action_means[forget_inds, i, :]
                else:
                    all_actions[:, i, :] = self.beta*(curr_action_means[:, i, :]) + (1-self.beta)*all_actions[:, i-1, :]
            all_actions = np.clip(all_actions, -1.0, 1.0)

            for n in range(0, self.num_samples_per_it, self.num_envs):
                done_any = np.zeros((self.num_envs,))
                inds = np.arange(n, n+self.num_envs)
                actions = all_actions[inds, ...]
                rewards = np.zeros((self.num_envs,))
                self.envs.set_states(env_states)
                for j in range(self.tau):
                    obs, rews, done, infos = self.envs.step(actions[:, j, :])
                    curr_states[inds, j, :] = obs
                    if self.sum_rewards:
                        if self.mask_done:
                            done_any = np.clip(done_any + done, 0, 1)
                            rewards += (1-done_any) * rews * self.reward_scaler[j]
                        else:
                            rewards += rews*self.reward_scaler[j]
                        try:
                            extra_rew = 1.0 * np.array([info["is_success"] for info in infos]) * self.reward_scaler[self.tau-1-j]
                        except:
                            extra_rew = np.zeros_like(rewards)
                        rewards += extra_rew
                    else:
                        if j == (self.tau - 1):
                            rewards = rews
                curr_rewards[inds] = rewards.copy()

            inds_prev = np.argsort(-curr_rewards[:num_prev])
            if len(inds_prev) > 0:
                if curr_rewards[inds_prev[0]] > best_reward:
                    best_reward = curr_rewards[inds_prev[0]].copy()
                    best_ac_traj = all_actions[inds_prev[0], :, :].copy()
                    best_ac_means = curr_action_means[inds_prev[0], :, :].copy()

                elite_inds_prev = inds_prev[:num_prev_best]

            inds_other = np.argsort(-curr_rewards[num_prev:])
            if len(inds_other) > 0:
                best_i = num_prev + inds_other[0]
                if curr_rewards[best_i] > best_reward:
                    best_reward = curr_rewards[best_i].copy()
                    best_ac_traj = all_actions[best_i, :, :].copy()
                    best_ac_means = curr_action_means[best_i, :, :].copy()

                elite_inds_other = inds_other[:num_other_best] + num_prev

            best_rewards_it.append(best_reward.copy())

            if it < self.num_iterations - 1:
                if len(inds_prev) > 0:
                    elite_ac_means_prev = curr_action_means[elite_inds_prev, ...]
                    past_actions_prev = past_actions[elite_inds_prev, :]
                    elite_ac_means_prev = np.tile(elite_ac_means_prev, (num_copies, 1, 1))[:num_prev, ...]
                    past_actions_prev = np.tile(past_actions_prev, (num_copies, 1))[:num_prev, :]
                    noise = (np.random.rand(*elite_ac_means_prev.shape) < self.later_noise_frac[it]).astype(int) * \
                            self.later_noise[it] * np.random.randn(*elite_ac_means_prev.shape)
                    curr_action_means[:num_prev, ...] = np.clip(elite_ac_means_prev+noise, -1.0, 1.0)
                    past_actions[:num_prev, :] = past_actions_prev.copy()

                if len(inds_other) > 0:
                    elite_ac_means_other = curr_action_means[elite_inds_other, ...]
                    past_actions_other = past_actions[elite_inds_other, :]
                    elite_ac_means_other = np.tile(elite_ac_means_other, (num_copies, 1, 1))[:num_other, ...]
                    past_actions_other = np.tile(past_actions_other, (num_copies, 1))[:num_other, :]
                    noise = (np.random.rand(*elite_ac_means_other.shape) < self.later_noise_frac[it]).astype(int) * \
                            self.later_noise[it] * np.random.randn(*elite_ac_means_other.shape)
                    curr_action_means[num_prev:, ...] = np.clip(elite_ac_means_other+noise, -1.0, 1.0)
                    past_actions[num_prev:, :] = past_actions_other.copy()

        return best_reward, best_ac_traj, best_ac_means, best_rewards_it, (np.mean(curr_rewards), np.std(curr_rewards))