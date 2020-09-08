import numpy as np

class Planner(object):
    def __init__(self, envs, name=None, init_noise=0.9):
        self.envs = envs
        self.num_envs = envs.num_envs
        self.ac_dim = self.envs.ac_dim
        self.init_noise = init_noise
        self.name = name

    def plan(self, env_states, prev_actions=None, prev_best_ac_mean=None, tau=15, num_searches=1,
             num_samples_per_search=1000, num_iterations=40, sync_searches_every=10,
             one_search_after=20, beta=0.7, tau_scaler=1.0, success_reward=True, min_noise=0.3, max_noise=0.4,
             min_noise_frac=0.3, max_noise_frac=0.4, frac_best=0.05):
        assert num_samples_per_search % self.envs.num_envs == 0
        reward_scaler = (tau_scaler)**(np.arange(tau)+1)
        if prev_actions is None:
            prev_actions = np.zeros((num_searches, self.ac_dim))
        curr_action_means = [np.clip(self.init_noise*np.random.randn(num_samples_per_search, tau, self.ac_dim), -1.0, 1.0) for _ in range(num_searches)]
        noise_0 = np.random.uniform(min_noise, max_noise, num_searches)
        noise_frac_0 = np.random.uniform(min_noise_frac, max_noise_frac, num_searches)
        if prev_best_ac_mean is not None:
            for i in range(num_searches):
                curr_action_means[i][:, :tau-1, :] = prev_best_ac_mean[np.newaxis, ...].copy()
                shape = curr_action_means[i][:, :tau-1,:].shape
                curr_action_means[i][:, :tau-1, :] += (np.random.rand(*shape) < noise_frac_0[i]).astype(int)*noise_0[i]*np.random.randn(*shape)
                curr_action_means[i][:, :tau-1, :] = np.clip(curr_action_means[i][:, :tau-1, :], -1.0, 1.0)

        past_actions = [np.tile(prev_actions[i,:], (num_samples_per_search,1)) for i in range(num_searches)]
        num_best = int(frac_best*num_samples_per_search)
        num_copies = num_samples_per_search // num_best

        best_ac_trajs = [None]*num_searches
        best_ac_means = [None]*num_searches
        best_rewards = [-np.inf]*num_searches
        best_rewards_it = [[]]*num_searches

        for it in range(num_iterations):
            for s in range(num_searches):
                curr_rewards = np.zeros((num_samples_per_search,))
                all_actions = np.zeros((num_samples_per_search, tau, self.ac_dim))
                for i in range(tau):
                    if i==0:
                        all_actions[:, i, :] = beta*curr_action_means[s][:, i, :] + (1-beta)*past_actions[s]
                    else:
                        all_actions[:, i, :] = beta*curr_action_means[s][:, i, :] + (1-beta)*all_actions[:, i-1, :]
                all_actions = np.clip(all_actions, -1.0, 1.0)

                for n in range(0, num_samples_per_search, self.num_envs):
                    inds = np.arange(n, n + self.num_envs)
                    actions = all_actions[inds, ...]
                    rewards = np.zeros((self.num_envs,))
                    self.envs.set_states(env_states)
                    for j in range(tau):
                        obs, rews, _, infos = self.envs.step(actions[:, j, :])
                        rewards += rews*reward_scaler[j]
                        if success_reward:
                            try:
                                extra_rew = 1.0*np.array([info["is_success"] for info in infos])*reward_scaler[tau-1-j]
                            except:
                                extra_rew = np.zeros_like(rewards)
                            rewards += extra_rew
                    curr_rewards[inds] = rewards.copy()

                inds = np.argsort(-curr_rewards)
                if curr_rewards[inds[0]] > best_rewards[s]:
                    best_rewards[s] = curr_rewards[inds[0]].copy()
                    best_ac_trajs[s] = all_actions[inds[0], :, :].copy()
                    best_ac_means[s] = curr_action_means[s][inds[0], ...].copy()
                best_rewards_it[s].append(best_rewards[s].copy())

                elite_inds = inds[:num_best]
                if it < num_iterations - 1:
                    elite_ac_means_prev = curr_action_means[s][elite_inds, ...]
                    past_actions_prev = past_actions[s][elite_inds, :]
                    elite_ac_means_prev = np.tile(elite_ac_means_prev, (num_copies, 1, 1))[:num_samples_per_search, ...]
                    past_actions_prev = np.tile(past_actions_prev, (num_copies, 1))[:num_samples_per_search, ...]
                    noise_val = np.random.uniform(min_noise, max_noise, 1)
                    noise_frac_val = np.random.uniform(min_noise_frac, max_noise_frac, 1)
                    noise = (np.random.rand(*elite_ac_means_prev.shape) < noise_frac_val).astype(int)*noise_val*np.random.randn(*elite_ac_means_prev.shape)
                    curr_action_means[s] = np.clip(elite_ac_means_prev + noise, -1.0, 1.0)
                    past_actions[s] = past_actions_prev.copy()
            if it % sync_searches_every == 0 and it > 0:
                if it >= one_search_after:
                    pass
                else:
                    best_ind = np.argmax(best_rewards)
                    for w in range(num_searches):
                        curr_action_means[w] = curr_action_means[best_ind].copy()
                        past_actions[w] = past_actions[best_ind].copy()
                        best_rewards[w] = best_rewards[best_ind].copy()
                        best_ac_means[w] = best_ac_means[best_ind].copy()
                        best_ac_trajs[w] = best_ac_trajs[best_ind].copy()
            if it == one_search_after:
                best_ind = np.argmax(best_rewards)
                best_rewards = [best_rewards[best_ind]]
                best_ac_trajs = [best_ac_trajs[best_ind]]
                best_ac_means = [best_ac_means[best_ind]]
                curr_action_means = [np.concatenate(curr_action_means, axis=0)]
                past_actions = [np.concatenate(past_actions, axis=0)]
                num_samples_per_search *= num_searches
                num_searches = 1
                num_best = int(frac_best * num_samples_per_search)
                num_copies = num_samples_per_search // num_best
        return best_rewards, best_ac_trajs, best_ac_means, best_rewards_it