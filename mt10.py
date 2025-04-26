import benchmark_evaluator
import metaworld

benchmark_evaluator.evaluate_benchmark(
    metaworld.MT10(), False, benchmark_evaluator.TrainingParameters(2e5, 32)
)
