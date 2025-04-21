import metaworld
import evaluate
import gymnasium
import random

ml10 = metaworld.MT10()
train_envs = {}

for name, env_cls in ml10.train_classes.items():
    env = env_cls()
    tasks = [task for task in ml10.train_tasks if task.env_name == name]
    # env = ml10.train_tasks.sample_task(name)
    if tasks:
        env.set_task(tasks[0])
        train_envs[name] = env

print(train_envs)

from stable_baselines3 import SAC
from Environment_Wrapper import MultiTaskWrapper

multi_task_env = MultiTaskWrapper(train_envs)

model = SAC("MlpPolicy", multi_task_env, verbose=1)
model.learn(total_timesteps=1e6)
model.save("sac_mt10_model")
evaluate.evaluate(
    lambda obs: model.predict(obs, deterministic=True)[0],
    multi_task_env,
    num_episodes=100,
    render=False,
)
