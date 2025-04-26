import benchmark_evaluator
import metaworld


def run_experiment(
    total_steps,
    batch_size,
    include_individual_metalearn,
    include_overall_multitask,
    include_overall_metalearn,
):
    total_steps = int(float(total_steps))
    mt10 = metaworld.MT10()
    all_test_classes = list(mt10.train_classes.keys())
    mt1_params = benchmark_evaluator.TrainingParameters(
        total_steps // 10,
        batch_size,
        None,
    )
    for test_class in all_test_classes:
        mt1 = metaworld.MT1(test_class)
        ml1 = metaworld.ML1(test_class)
        print("Evaluating class: ", test_class)
        print("Evaluating MT1")
        benchmark_evaluator.evaluate_benchmark(
            mt1,
            False,
            mt1_params,
            10,
            "_".join(
                [
                    "sac",
                    "mt1",
                    "default",
                    test_class,
                    str(total_steps // 10),
                    str(batch_size),
                ]
            ),
        )
        if include_individual_metalearn:
            print("Evaluating ML1")
            benchmark_evaluator.evaluate_benchmark(
                ml1,
                True,
                mt1_params,
                10,
                "_".join(
                    [
                        "sac",
                        "metalearn1",
                        "default",
                        test_class,
                        str(total_steps // 10),
                        str(batch_size),
                    ]
                ),
            )
    if include_overall_multitask:
        print("Evaluating MT10")
        mt10_params = benchmark_evaluator.TrainingParameters(
            total_steps, batch_size, None
        )
        benchmark_evaluator.evaluate_benchmark(
            mt10,
            False,
            mt10_params,
            100,
            "_".join(["sac", "mt10", "default", str(total_steps), str(batch_size)]),
        )
    if include_overall_metalearn:
        print("Evaluating ML10")
        ml10 = metaworld.ML10()
        benchmark_evaluator.evaluate_benchmark(
            ml10,
            True,
            mt10_params,
            100,
            "_".join(
                ["sac", "metalearn10", "default", str(total_steps), str(batch_size)]
            ),
        )
