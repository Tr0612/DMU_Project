import ml10_experiments

for steps in [1, 1e4, 2e4, 8e4, 2e5, 4e5, 8e5, 2e6]:
    ml10_experiments.run_experiment(steps, [10, 20, 40, 80, 160], 32, [400, 400])
