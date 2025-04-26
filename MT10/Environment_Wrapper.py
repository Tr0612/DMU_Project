from gymnasium import spaces
import numpy as np
import gymnasium as gym
import metaworld

class MultiTaskWrapper(gym.Env):
    def __init__(self,env_dict):
        super().__init__()
        # self.benchmark = benchmark
        self.envs = list(env_dict.values())
        self.task_names = list(env_dict.keys())
        self.num_tasks = len(self.envs)
        self.current_task = 0
        
        
        self.observation_space = spaces.Box(
            low = np.inf,
            high = np.inf,
            shape = (self.envs[0].observation_space.shape[0]+self.num_tasks,),
            dtype=np.float32
        )
        self.action_space = self.envs[0].action_space
        
    
    def reset(self,*, seed=None, options=None):
        self.current_task = np.random.randint(self.num_tasks)
        self.env = self.envs[self.current_task]
        obs,_ = self.env.reset()
        return self._concat_obs(obs), {}
    
    def step(self,action):
        obs,reward,done,truncated,info = self.env.step(action)
        return self._concat_obs(obs),reward,done,truncated,info
    
    def _concat_obs(self,obs):
        one_hot = np.zeros(self.num_tasks)
        one_hot[self.current_task] = 1.0
        return np.concatenate([obs,one_hot])

class MetaWorldRlib(gym.Env):
    def __init__(self):
        super(MetaWorldRlib, self).__init__()

        self.ml10 = metaworld.MT10()
        self.env_dict = {}
        
        for name, env_cls in self.ml10.train_classes.items():
            env = env_cls()
            task = [task for task in self.ml10.train_tasks if task.env_name == name][0]
            env.set_task(task)
            self.env_dict[name] = env

        self.env_names = list(self.env_dict.keys())
        self.num_tasks = len(self.env_names)
        self.current_task_idx = 0
        self.current_env = self.env_dict[self.env_names[self.current_task_idx]]

        obs_dim = self.current_env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim + self.num_tasks,), dtype=np.float32
        )
        self.action_space = self.current_env.action_space

    def reset(self, *, seed=None, options=None):
        self.current_task_idx = np.random.randint(0, self.num_tasks)
        self.current_env = self.env_dict[self.env_names[self.current_task_idx]]
        obs, _ = self.current_env.reset()
        return self._concat_obs(obs), {}

    def step(self, action):
        obs, reward, done, truncated, info = self.current_env.step(action)
        return self._concat_obs(obs), reward, done, truncated, info

    def _concat_obs(self, obs):
        one_hot = np.zeros(self.num_tasks, dtype=np.float32)
        one_hot[self.current_task_idx] = 1.0
        return np.concatenate([obs, one_hot])