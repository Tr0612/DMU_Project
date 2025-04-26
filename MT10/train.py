from stable_baselines3 import SAC
from Environment_Wrapper import MultiTaskWrapper,MetaWorldRlib
from CustomSAC import CustomSAC
from Prioritized_Wrapper import PrioritizedReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.sac import SACConfig
from ray import tune
import metaworld

def env_creator(config):
    return MetaWorldRlib()

register_env("MetaWorldMT10",env_creator)

config = (
    SACConfig()
    .environment(env="MetaWorldMT10")
    .framework("torch")
    .resources(
        num_gpus=1
    )
    .env_runners(num_env_runners=1)  # new API
    .training(
        replay_buffer_config={
            "type": "PrioritizedEpisodeReplayBuffer",  # ✅ very important
            "capacity": 5000,
            "alpha": 0.6,
            "beta": 0.4,
        },
        actor_lr=3e-4,
        critic_lr=3e-4,
        train_batch_size_per_learner=256,
        gamma=0.99,
        tau=0.005,
    )
)

tune.Tuner(
    "SAC",
    param_space=config.to_dict(),
    run_config=tune.RunConfig(
        stop={"timesteps_total": 1_000_000},
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=10,
            checkpoint_at_end=True,
        ),
    ),
).fit()



'''SB3 Setup'''

# ml10 = metaworld.MT10()
# train_envs = {}

# for name,env_cls in ml10.train_classes.items():
#     env = env_cls()
#     tasks = [task for task in ml10.train_tasks
#                         if task.env_name == name]
#     # env = ml10.train_tasks.sample_task(name)
#     if tasks:
#         env.set_task(tasks[0])
#         train_envs[name] = env

# multi_task_env = MultiTaskWrapper(train_envs)

# env = DummyVecEnv([lambda: multi_task_env])


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

# model = CustomSAC(
#     "MlpPolicy",
#     env,
#     replay_buffer_class=PrioritizedReplayBuffer,
#     replay_buffer_kwargs={"alpha": 0.6},
#     verbose=1,
#     batch_size=256,
#     buffer_size=1_000_000,
#     learning_rate=3e-4,
#     learning_starts=10000,
#     tau=0.005,
#     gamma=0.99,
#     train_freq=1,
#     gradient_steps=1,
#     target_update_interval=1
# )



# model.learn(total_timesteps=1_000_000)
# model.save("sac_mt10_model_per")