import evaluate
from stable_baselines3 import SAC
from Environment_Wrapper import MultiTaskWrapper


class TrainingParameters:
    def __init__(self, timesteps, batch_size):
        self.timesteps = timesteps
        self.batch_size = batch_size


def train_on_benchmark(benchmark, benchmark_name, parameters: TrainingParameters):
    train_envs = {}

    for name, env_cls in benchmark.train_classes.items():
        env = env_cls()
        tasks = [task for task in benchmark.train_tasks if task.env_name == name]
        # env = ml10.train_tasks.sample_task(name)
        if tasks:
            env.set_task(tasks[0])
            train_envs[name] = env

    print(train_envs)

    multi_task_env = MultiTaskWrapper(train_envs)

    model = SAC(
        "MlpPolicy", multi_task_env, batch_size=parameters.batch_size, verbose=1
    )
    model.learn(total_timesteps=parameters.timesteps, progress_bar=True)
    model.save("sac_" + benchmark_name + "_model")
    evaluation = evaluate.evaluate(
        lambda obs: model.predict(obs, deterministic=True)[0],
        multi_task_env,
        num_episodes=100,
        render=False,
    )
    return model, evaluation
