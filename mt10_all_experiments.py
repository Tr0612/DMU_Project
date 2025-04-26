import benchmark_evaluator
import metaworld

def run_experiment(total_steps, batch_size):
    mt10 = metaworld.MT10()
    all_test_classes = list(mt10.test_classes().keys())
    mt1_params = benchmark_evaluator.TrainingParameters(total_steps / 10, batch_size)
    for test_class in all_test_classes:
        mt1 = metaworld.MT1(test_class)
        ml1 = metaworld.ML1(test_class)
        benchmark_evaluator.evaluate_benchmark(
            mt1,
            False,
            mt1_params,
            "_".join("sac", "mt1", test_class, str(total_steps / 10), str(batch_size)),
        )
        benchmark_evaluator.evaluate_benchmark(
            ml1,
            True,
            mt1_params,
            "_".join(
                "sac", "metalearn1", test_class, str(total_steps / 10), str(batch_size)
            ),
        )
    mt10_params = benchmark_evaluator.TrainingParameters(total_steps, batch_size)
    benchmark_evaluator.evaluate_benchmark(
        mt10,
        False,
        mt10_params,
        "_".join("sac", "mt10", str(total_steps), str(batch_size)),
    )
    ml10 = metaworld.ML10()
    benchmark_evaluator.evaluate_benchmark(
        ml10,
        True,
        mt10_params,
        "_".join("sac", "metalearn10", str(total_steps), str(batch_size)),
    )
