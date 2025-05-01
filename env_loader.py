from environment_wrapper import MultiTaskWrapper

def environment_with_task(env, task):
    env.set_task(task)
    return env

def load_train_envs_from_benchmark(benchmark):
    train_envs = {}
    for name, env_cls in benchmark.train_classes.items():
        tasks = [task for task in benchmark.train_tasks if task.env_name == name]
        envs = [environment_with_task(env_cls(), task) for task in tasks]
        train_envs[name] = envs
    return train_envs

def load_test_envs_from_benchmark(benchmark):
    test_envs = {}
    for name, env_cls in benchmark.test_classes.items():
        tasks = [task for task in benchmark.test_tasks if task.env_name == name]
        envs = [environment_with_task(env_cls(), task) for task in tasks]
        test_envs[name] = envs
    return test_envs

def multitask_env_from_benchmark(benchmark):
    train_envs = load_train_envs_from_benchmark(benchmark)
    return MultiTaskWrapper(train_envs, {})

def metalearning_env_from_benchmark(benchmark):
    train_envs = load_train_envs_from_benchmark(benchmark)
    test_envs = load_test_envs_from_benchmark(benchmark)
    return MultiTaskWrapper(train_envs, test_envs)
