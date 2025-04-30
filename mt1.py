import benchmark_evaluator
import metaworld

# reach_mt1 = metaworld.MT1("window-close-v2")

model, evaluation = benchmark_evaluator.evaluate_benchmark(
    metaworld.MT1("window-close-v2"),
    False,
    benchmark_evaluator.TrainingParameters(2e4, 32, None, [400, 400]),
    10,
    None,
)