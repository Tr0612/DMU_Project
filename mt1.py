import benchmark_evaluator
import metaworld

reach_mt1 = metaworld.MT1("reach-v2")

model, evaluation = benchmark_evaluator.evaluate_benchmark(
    metaworld.MT1("reach-v2"),
    False,
    benchmark_evaluator.TrainingParameters(1e4, 32, None),
    10,
    "sac-reach-her-v2-1e4-32",
)

model.replay_buffer
