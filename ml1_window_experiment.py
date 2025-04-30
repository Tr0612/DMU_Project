import ml1_experiments

for steps in [5e3, 1e4, 3e4, 9e4, 3e5]:
    ml1_experiments.run_experiment(steps, 32, [400, 400], ["window-close-v2"])
