import ml10_experiments

for buffer in [None, "per"]:
    for steps in [1, 1e4, 2e4, 4e4, 8e4, 2e5, 4e5, 8e5]:
        ml10_experiments.run_experiment(
            steps, [10, 20, 30, 40, 60, 80, 120, 160], 32, [400, 400], buffer
        )
