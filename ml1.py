import benchmark_evaluator
import metaworld

reach_ml1 = metaworld.ML1("window-close-v2")

model, evaluation = benchmark_evaluator.evaluate_benchmark(
    reach_ml1,
    True,
    benchmark_evaluator.TrainingParameters(1e4, 32, None, [400, 400]),
    200,
    None,
)

# model.replay_buffer
