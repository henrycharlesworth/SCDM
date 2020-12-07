from gym.envs.robotics.hand.manipulate import ManipulateEnv
from gym import utils
from gym.envs.robotics import rotations
import numpy as np
import os

class HandPen(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='random', target_rotation='xyz', reward_type='sparse',
                 dist_fact=10.0, alpha=1.0):
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        self.goal_dim = 7
        self.state_dim = 61
        self.ac_dim = 20
        self.death_tol = 0.0
        self.alpha = alpha
        self.dist_fact = dist_fact
        ManipulateEnv.__init__(self,
                               model_path=os.path.join('hand', 'manipulate_pen.xml'), target_position=target_position,
                               target_rotation=target_rotation,
                               target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
                               reward_type=reward_type, randomize_initial_rotation=False)
        self.env = self

    def get_current_reward(self):
        obs = self._get_obs()
        return self.compute_reward(obs["achieved_goal"], self.goal, None)

    def set_goal(self, goal):
        self.goal = goal

    def save_state(self):
        return self.sim.get_state()

    def load_state(self, state):
        self.sim.set_state(state)
        self.sim.forward()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        is_dead = False
        if obs["achieved_goal"][2] < self.death_tol:
            is_dead = True

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'is_dead': is_dead
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, goal, info):
        alpha = self.alpha
        d_pos, d_rot = self._goal_distance(achieved_goal, goal)
        dist = self.dist_fact * d_pos + d_rot
        if achieved_goal[2] < self.death_tol:
            prefactor = 0.0
        else:
            prefactor = 1.0
        return prefactor * np.exp(-alpha * dist)