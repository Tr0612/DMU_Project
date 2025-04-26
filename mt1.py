import benchmark_evaluator
import metaworld

reach_mt1 = metaworld.MT1("reach-v2")

benchmark_evaluator.evaluate_benchmark(
    metaworld.MT1("reach-v2"),
    False,
    benchmark_evaluator.TrainingParameters(1e4, 32),
    "sac-reach-v2-1e4-32",
)
