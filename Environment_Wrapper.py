from gymnasium import spaces
import numpy as np
import gymnasium as gym


class MultiTaskWrapper(gym.Env):
    def __init__(self, env_dict):
        super().__init__()
        # self.benchmark = benchmark
        self.envs = list(env_dict.values())
        self.task_names = list(env_dict.keys())
        self.num_tasks = len(self.envs)
        self.current_task = 0

        self.observation_space = spaces.Box(
            low=np.concatenate([self.envs[0].observation_space.low, np.zeros(self.num_tasks)]),
            high=np.concatenate([self.envs[0].observation_space.high, np.ones(self.num_tasks)]),
            shape=(self.envs[0].observation_space.shape[0] + self.num_tasks,),
            dtype=np.float32,
        )
        self.action_space = self.envs[0].action_space

    def set_render_mode(self, render_mode):
        for env in self.envs:
            env.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        self.current_task = np.random.randint(self.num_tasks)
        self.env = self.envs[self.current_task]
        obs, _ = self.env.reset()
        return self._concat_obs(obs), {}

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if truncated:
            self.reset()
        else:
            info["task_name"] = self.task_names[self.current_task]
        return self._concat_obs(obs), reward, done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        for env in self.envs:
            env.close()

    def _concat_obs(self, obs):
        one_hot = np.zeros(self.num_tasks)
        one_hot[self.current_task] = 1.0
        return np.concatenate([obs, one_hot])
