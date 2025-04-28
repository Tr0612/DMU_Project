import mt10_all_experiments

for steps in [2e5, 5e5, 1e6, 3e6]:
    mt10_all_experiments.run_experiment(steps, 32, [400, 400], False, False, False)
