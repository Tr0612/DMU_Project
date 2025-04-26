import evaluate
import env_loader
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback


class TrainingParameters:
    def __init__(self, timesteps, batch_size, replay_buffer_type):
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.replay_buffer_type = replay_buffer_type

class SACCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=1)

    def _on_step(self):
        # print("Reward: ", self.model.ep_info_buffer)
        return super()._on_step()


def evaluate_benchmark(
    benchmark,
    is_meta_learning,
    parameters: TrainingParameters,
    evaluation_episodes,
    saved_model_name=None,
):
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
        else:
            replay_buffer_class = None
        callback = SACCallback()
        model = SAC(
            "MlpPolicy",
            env,
            replay_buffer_class=replay_buffer_class,
            batch_size=parameters.batch_size,
            callback=callback,
            verbose=1,
        )
        env.enter_train_mode()
        model.learn(total_timesteps=parameters.timesteps, progress_bar=True)
        if saved_model_name:
            model.save(saved_model_name)
    if is_meta_learning:
        env.enter_test_mode()
    evaluation = evaluate.evaluate(
        lambda obs: model.predict(obs, deterministic=True)[0],
        env,
        num_episodes=evaluation_episodes,
        render=False,
    )
    if saved_model_name:
        with open(saved_model_name + "_results.txt", "w") as file:
            file.write(str(evaluation))
    return model, evaluation
