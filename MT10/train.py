from stable_baselines3 import SAC
from Environment_Wrapper import MultiTaskWrapper


multi_task_env = MultiTaskWrapper(train_envs)

model = SAC("MlpPolicy",multi_task_env,verbose=1)
model.learn(total_timesteps=1_000_00)
model.save("sac_mt10_model")