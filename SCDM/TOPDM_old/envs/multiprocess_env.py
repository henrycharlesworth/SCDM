import numpy as np
import copy
from multiprocessing import Process, Pipe
from SCDM.TOPDM.envs.util import CloudpickleWrapper

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    env.reset()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if isinstance(ob, dict):
                remote.send((ob["observation"], reward, done, info))
            else:
                remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob["observation"])
        elif cmd == 'get_state':
            state = env.env.sim.get_state()
            remote.send(CloudpickleWrapper(state))
        elif cmd == 'set_state':
            env.env.sim.set_state(data.x)
            env.env.sim.forward()
            remote.send(True) #confirm
        elif cmd == 'set_goal':
            env.goal = data
            env.env.goal = data
            remote.send(True)
        elif cmd == 'get_current_reward':
            obs = env.env._get_obs()
            try:
                rew = env.env.compute_reward(obs["achieved_goal"], env.env.goal, None)
            except:
                rew = env.env.compute_reward() #pen spin
            remote.send(rew)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class SubprocVecEnv(object):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.num_envs = len(env_fns)
        self.observation_space = observation_space
        self.action_space = action_space
        self.dummy = env_fns[0]()
        self.ac_dim = self.dummy.action_space.shape[0]
        try:
            self.goal_dim = self.dummy.unwrapped.goal.shape[0]
        except:
            self.goal_dim = self.dummy.unwrapped.goal["object_1"].shape[0]*2
        try:
            self.state_dim = self.dummy.observation_space["observation"].shape[0]
        except:
            self.state_dim = self.dummy.observation_space.shape[0]

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self):
        print('Render not defined for %s' % self)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def get_states_async(self):
        for remote in self.remotes:
            remote.send(('get_state', None))
        self.waiting = True

    def get_states_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return results

    def get_states(self):
        self.get_states_async()
        return self.get_states_wait()

    def set_states(self, states):
        for remote, state in zip(self.remotes, states):
            remote.send(('set_state', state))
        return [remote.recv() for remote in self.remotes]

    def set_goals(self, goal):
        goals = [copy.copy(goal) for _ in range(self.num_envs)]
        for remote, goal in zip(self.remotes, goals):
            remote.send(('set_goal', goal))
        return [remote.recv() for remote in self.remotes]

    def get_current_rewards(self):
        for remote in self.remotes:
            remote.send(('get_current_reward', None))
        return [remote.recv() for remote in self.remotes]

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True