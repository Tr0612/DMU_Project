from gymnasium import spaces
import numpy as np
import gymnasium as gym


class MultiTaskWrapper(gym.Env):
    def __init__(self, train_env_dict, test_env_dict, include_skill_id=True):
        super().__init__()
        self.mode = "train"
        train_envs = []
        test_envs = []
        skill_names = []
        unique_skill_names = []
        for name in train_env_dict.keys():
            for train_env in train_env_dict[name]:
                train_envs.append(train_env)
                skill_names.append(name)
                if include_skill_id and name not in unique_skill_names:
                    unique_skill_names.append(name)
        for name in test_env_dict.keys():
            for test_env in test_env_dict[name]:
                test_envs.append(test_env)
                skill_names.append(name)
                if include_skill_id and name not in unique_skill_names:
                    unique_skill_names.append(name)

        self.envs = train_envs + test_envs
        self.skill_names = skill_names
        self.unique_skill_names = unique_skill_names
        self.num_train_tasks = len(train_envs)
        self.num_test_tasks = len(test_envs)

        self.current_task = 0

        total_num_tasks = self.num_train_tasks + self.num_test_tasks

        self.observation_space = spaces.Box(
            low=np.concatenate(
                [
                    self.envs[0].observation_space.low,
                    np.zeros(total_num_tasks + len(self.unique_skill_names)),
                ]
            ),
            high=np.concatenate(
                [
                    self.envs[0].observation_space.high,
                    np.ones(total_num_tasks + len(self.unique_skill_names)),
                ]
            ),
            shape=(
                self.envs[0].observation_space.shape[0]
                + total_num_tasks
                + len(self.unique_skill_names),
            ),
            dtype=np.float32,
        )
        self.action_space = self.envs[0].action_space

    def enter_train_mode(self):
        self.mode = "train"

    def enter_test_mode(self):
        self.mode = "test"

    def random_task_index(self):
        if self.mode == "train":
            return np.random.randint(self.num_train_tasks)
        elif self.mode == "test":
            return self.num_train_tasks + np.random.randint(self.num_test_tasks)
        else:
            print("Unrecognized mode")
            raise Exception()

    def set_render_mode(self, render_mode):
        for env in self.envs:
            env.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        self.current_task = self.random_task_index()
        self.env = self.envs[self.current_task]
        obs, _ = self.env.reset()
        return self._concat_obs(obs), {}

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if truncated or done:
            self.reset()
        else:
            info["task_name"] = self.skill_names[self.current_task]
        return self._concat_obs(obs), reward, done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        for env in self.envs:
            env.close()

    def _concat_obs(self, obs):
        one_hot_task = np.zeros(self.num_train_tasks + self.num_test_tasks)
        one_hot_task[self.current_task] = 1.0
        one_hot_skill = np.zeros(len(self.unique_skill_names))
        if len(self.unique_skill_names) > 0:
            skill_name = self.skill_names[self.current_task]
            index_of_skill_name = self.unique_skill_names.index(skill_name)
            one_hot_skill[index_of_skill_name] = 1.0
        return np.concatenate([obs, one_hot_task, one_hot_skill])
