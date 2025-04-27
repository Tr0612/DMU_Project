import mt10_all_experiments

for steps in [5e4, 1e5, 2e5, 5e5, 1e6]:
    mt10_all_experiments.run_experiment(steps, 32, True, False, False)
