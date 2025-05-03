import ml10_experiments

for steps in [1, 1e4, 8e4, 2e5, 8e5]:
    ml10_experiments.run_experiment(steps, [10, 20, 40, 80, 160, 320, 480], 32, [400, 400])
