import benchmark_evaluator
import metaworld

window_open_ml1 = metaworld.ML1("window-open-v2")

print("Evaluating ML")
model, evaluation = benchmark_evaluator.evaluate_benchmark(
    window_open_ml1,
    True,
    benchmark_evaluator.TrainingParameters(1e3, 32, None, [400, 400]),
    15,
    None,
)

# model.replay_buffer
print("Evaluating MT")
window_open_mt1 = metaworld.MT1("window-open-v2")
model, evaluation = benchmark_evaluator.evaluate_benchmark(
    window_open_mt1,
    False,
    benchmark_evaluator.TrainingParameters(1e5, 32, None, [400, 400]),
    10,
    None,
)
