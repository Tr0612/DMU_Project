import evaluate
import env_loader
import numpy as np
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn as nn
from Prioritized_Wrapper import PrioritizedReplayBuffer


class TrainingParameters:
    def __init__(
        self,
        timesteps,
        batch_size,
        replay_buffer_type,
        architecture,
        train_during_testing=False,
    ):
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.replay_buffer_type = replay_buffer_type
        self.architecture = architecture
        self.train_during_testing = train_during_testing


class SACCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=1)

    def _on_step(self):
        # print("Reward: ", self.model.ep_info_buffer)
        return super()._on_step()


def handle_online_step(model, obs, next_obs, action, reward, done, info):
    model.replay_buffer.add(
        np.array([obs]),
        np.array([next_obs]),
        np.array([action]),
        np.array([reward]),
        np.array([done]),
        [info],
    )
    model.train(gradient_steps=model.gradient_steps, batch_size=model.batch_size)

def perform_evaluation(
    model,
    env,
    evaluation_episodes,
    evaluation_max_steps,
    on_step,
    results_name,
    results_suffix="",
):
    evaluation = evaluate.evaluate(
        lambda obs: model.predict(obs, deterministic=True)[0],
        env,
        num_episodes=int(evaluation_episodes),
        max_steps_per_episode=evaluation_max_steps,
        render=False,
        on_step=on_step,
    )
    if results_name:
        with open(results_name + "_" + results_suffix + "_results.txt", "w") as file:
            file.write(str(evaluation))
    return evaluation


def evaluate_benchmark(
    benchmark,
    is_meta_learning,
    parameters: TrainingParameters,
    evaluation_episodes,
    evaluation_max_steps=1000,
    checkpoint_frequency=100,
    saved_model_name=None,
    results_name=None,
):
    if saved_model_name:
        if results_name is None:
            results_name = saved_model_name
    if not isinstance(evaluation_episodes, list):
        evaluation_episodes = [evaluation_episodes]
    if is_meta_learning:
        env = env_loader.metalearning_env_from_benchmark(benchmark)
    else:
        env = env_loader.multitask_env_from_benchmark(benchmark)
    try:
        model = SAC.load(saved_model_name, env)
        print("Successfully loaded trained model from disk")
    except:
        print("Training new model")
        if parameters.replay_buffer_type == "her":
            replay_buffer_class = HerReplayBuffer
        elif parameters.replay_buffer_type == "per":
            replay_buffer_class = PrioritizedReplayBuffer
        else:
            replay_buffer_class = None
        callback = SACCallback()
        policy_kwargs = {
            "net_arch": {"pi": parameters.architecture, "qf": parameters.architecture},
            "activation_fn": nn.ReLU,
        }
        model = SAC(
            "MlpPolicy",
            env,
            replay_buffer_class=replay_buffer_class,
            batch_size=parameters.batch_size,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )
        remaining_timesteps = parameters.timesteps
        if checkpoint_frequency is None:
            checkpoint_frequency = remaining_timesteps
        while True:
            env.enter_train_mode()
            model.learn(
                total_timesteps=checkpoint_frequency,
                callback=callback,
                progress_bar=True,
            )
            remaining_timesteps -= checkpoint_frequency
            if remaining_timesteps <= 0:
                break
            if is_meta_learning:
                env.enter_test_mode()
            perform_evaluation(
                model,
                env,
                evaluation_episodes[-1],
                evaluation_max_steps,
                None,
                results_name,
                str(parameters.timesteps - remaining_timesteps),
            )
        if saved_model_name:
            model.save(saved_model_name)
    on_step = None
    if is_meta_learning:
        env.enter_test_mode()
        if parameters.train_during_testing:
            on_step = (
                lambda obs, next_obs, action, reward, done, info: handle_online_step(
                    model, obs, next_obs, action, reward, done, info
                )
            )
    latest_evaluation = None
    for evaluation_episode in evaluation_episodes:
        latest_evaluation = perform_evaluation(
            model,
            env,
            evaluation_episode,
            evaluation_max_steps,
            on_step,
            results_name,
            "final_" + str(evaluation_episode),
        )
    return model, latest_evaluation
