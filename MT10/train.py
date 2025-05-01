from stable_baselines3 import SAC
from Environment_Wrapper import MultiTaskWrapper,MetaWorldRlib
from CustomSAC import CustomSAC
from Prioritized_Wrapper import PrioritizedReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.sac import SACConfig
from ray import tune
import metaworld

import matplotlib.pyplot as plt

def plot_priority_distribution(replay_buffer):
    priorities = replay_buffer.priorities
    priorities = priorities[:replay_buffer.pos]
    plt.figure(figsize=(8,4))
    plt.hist(priorities, bins=50, log=True)
    plt.title('Replay Buffer Priority Distribution')
    plt.xlabel('Priority')
    plt.ylabel('Frequency (log scale)')
    plt.grid(True)
    plt.savefig("priority_distribution_woper.png")
    plt.show()


# def env_creator(config):
#     return MetaWorldRlib()

# register_env("MetaWorldMT10",env_creator)

# config = (
#     SACConfig()
#     .environment(env="MetaWorldMT10")
#     .framework("torch")
#     .resources(
#         num_gpus=1
#     )
#     .env_runners(num_env_runners=1)  # new API
#     .training(
#         replay_buffer_config={
#             "type": "PrioritizedEpisodeReplayBuffer",  # ✅ very important
#             "capacity": 5000,
#             "alpha": 0.6,
#             "beta": 0.4,
#         },
#         actor_lr=3e-4,
#         critic_lr=3e-4,
#         train_batch_size_per_learner=256,
#         gamma=0.91,
#         tau=0.005,
#     )
# )

# tune.Tuner(
#     "SAC",
#     param_space=config.to_dict(),
#     run_config=tune.RunConfig(
#         stop={"timesteps_total": 1_000_000},
#         checkpoint_config=tune.CheckpointConfig(
#             checkpoint_frequency=10,
#             checkpoint_at_end=True,
#         ),
#     ),
# ).fit()



'''SB3 Setup'''

ml10 = metaworld.MT10()
mt1 = metaworld.MT1('reach-v2',seed =42)

env = mt1.train_classes['reach-v2']()
env.set_task(mt1.train_tasks[0])

train_envs = {}

for name,env_cls in ml10.train_classes.items():
    env = env_cls()
    tasks = [task for task in ml10.train_tasks
                        if task.env_name == name]
    # env = ml10.train_tasks.sample_task(name)
    if tasks:
        env.set_task(tasks[0])
        train_envs[name] = env

multi_task_env = MultiTaskWrapper(train_envs)

env = DummyVecEnv([lambda: multi_task_env])


# model = SAC("MlpPolicy",
#             env,
#             replay_buffer_class=PrioritizedReplayBuffer,
#             replay_buffer_kwargs={"alpha":0.6},
#             verbose=1,
#             batch_size =256,
#             buffer_size=1_000_000,
#             learning_rate=3e-4,
#             learning_starts=10000,
#             tau=0.005,
#             gamma=0.99,
#             train_freq=1,
#             gradient_steps=1,
#             target_update_interval=1
#             )

model = CustomSAC(
    "MlpPolicy",
    env,
    replay_buffer_class=PrioritizedReplayBuffer,
    replay_buffer_kwargs={"alpha": 0.6},
    verbose=1,
    batch_size=256,
    buffer_size=1_000_000,
    learning_rate=3e-4,
    learning_starts=10000,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=1
)

# model = CustomSAC(
#     policy="MlpPolicy",
#     env=env,
#     verbose=1,
#     batch_size=256,
#     learning_rate=3e-4,
#     buffer_size=1_000_000,
#     learning_starts=10000,
#     gamma=0.99,
#     tau=0.005,
#     train_freq=1,
#     gradient_steps=1,
#     target_update_interval=1,
# )

model.learn(total_timesteps=1000000)

# 4. Save model
model.save("sac_mt10_PER")


# model.learn(total_timesteps=5_00_00)
# model.save("sac_mt1_model_per")

print(model.replay_buffer.priorities)
plot_priority_distribution(model.replay_buffer)
# model = CustomSAC.load("sac_mt1_reach_model_vanilla")
# success = 0
# episodes = 20

# for ep in range(episodes):
#     obs, _ = env.reset()
#     done, truncated = False, False
#     while not (done or truncated):
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, truncated, info = env.step(action)
#         if info.get("success", 0) == 1:
#             success += 1
#             break

# print(f"Success rate: {success / episodes * 100:.1f}%")
































































# bhu