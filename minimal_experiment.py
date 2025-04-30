import mt10_all_experiments

for steps in [5e3, 1e4]:
    mt10_all_experiments.run_experiment(steps, 32, [400, 400], True, False, True, False)
