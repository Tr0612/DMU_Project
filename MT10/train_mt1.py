from stable_baselines3 import SAC
from Environment_Wrapper import MultiTaskWrapper,MetaWorldRlib
from CustomSAC import CustomSAC
from Prioritized_Wrapper import PrioritizedReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.sac import SACConfig
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv
from ray import tune
import metaworld
import os

mt1_tasks = [
    # 'reach-v2',
    # 'pick-place-v2',
    # 'peg-insert-side-v2',
    # 'window-close-v2'
    'window-open-v2', 
'drawer-open-v2', 
'button-press-topdown-v2',
'door-open-v2'
]


# 1) Stop when success ≥ 0.8 
timesteps = 100_000
save_dir = "./mt1_models/"
os.makedirs(save_dir, exist_ok=True)

# === Loop over all MT1 tasks ===
for task_name in mt1_tasks:
    print(f"\n🛠️ Training on task: {task_name}")

    # 1. Load MT1 environment
    mt1 = metaworld.MT1(task_name, seed=42)
    env = mt1.train_classes[task_name]()
    env.set_task(mt1.train_tasks[0])

    # Optional: Wrap with DummyVecEnv
    env = DummyVecEnv([lambda: env])
    eval_env = env
    
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=500,
        verbose=1
    )


    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=10_000,       
        n_eval_episodes=20,
        best_model_save_path=None,
        verbose=1,
    )

    # 2. Create SAC model
    model = CustomSAC(
        policy="MlpPolicy",
        replay_buffer_class=PrioritizedReplayBuffer,
        replay_buffer_kwargs={"alpha":0.6},
        env=env,
        verbose=1,
        batch_size=256,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
    )

    # model.set_device('cuda')
    # 3. Train
    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback)

    # 4. Save
    model_path = os.path.join(save_dir, f"sac_{task_name}_per_{timesteps}.zip")
    model.save(model_path)
    print(f"✅ Saved model to {model_path}")



for task_name in mt1_tasks:
    print(f"\n🛠️ Training on task: {task_name}")

    # 1. Load MT1 environment
    mt1 = metaworld.MT1(task_name, seed=42)
    env = mt1.train_classes[task_name]()
    env.set_task(mt1.train_tasks[0])
    
    env = DummyVecEnv([lambda: env])
    eval_env = env
    
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=500,
        verbose=1
    )


    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=10_000,       
        n_eval_episodes=20,
        best_model_save_path=None,
        verbose=1,
    )

    # 2. Create SAC model
    model = CustomSAC(
        policy="MlpPolicy",
        # replay_buffer_class=PrioritizedReplayBuffer,
        # replay_buffer_kwargs={"alpha":0.6},
        env=env,
        verbose=1,
        batch_size=256,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
    )

    # 3. Train
    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback)

    # 4. Save
    model_path = os.path.join(save_dir, f"sac_{task_name}_vanilla_{timesteps}.zip")
    model.save(model_path)
    print(f"✅ Saved model to {model_path}")

print("\n🎯 Training completed for all MT1 tasks!")

