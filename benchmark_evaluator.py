import evaluate
import env_loader
from stable_baselines3 import SAC

class TrainingParameters:
    def __init__(self, timesteps, batch_size):
        self.timesteps = timesteps
        self.batch_size = batch_size


def evaluate_benchmark(benchmark, is_meta_learning, parameters: TrainingParameters, saved_model_name=None):
    if is_meta_learning:
        env = env_loader.metalearning_env_from_benchmark(benchmark)
    else:
        env = env_loader.multitask_env_from_benchmark(benchmark)
    try:
        model = SAC.load(saved_model_name, env)
        print("Successfully loaded trained model from disk")
    except:
        print("Training new model")
        model = SAC("MlpPolicy", env, batch_size=parameters.batch_size, verbose=1)
        env.enter_train_mode()
        model.learn(total_timesteps=parameters.timesteps, progress_bar=True)
        if saved_model_name:
            model.save(saved_model_name)
    if is_meta_learning:
        env.enter_test_mode()
    evaluation = evaluate.evaluate(
        lambda obs: model.predict(obs, deterministic=True)[0],
        env,
        num_episodes=100,
        render=False,
    )
    if saved_model_name:
        with open(saved_model_name + "_results.txt", "w") as file:
            file.write(str(evaluation))
    return model, evaluation
