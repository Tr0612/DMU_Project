import benchmark_evaluator
import metaworld


def list_all_classes():
    mt10 = metaworld.MT10()
    return list(mt10.train_classes.keys())

def run_experiment(
    total_steps,
    batch_size,
    architecture,
    all_test_classes,
    evaluate_ml_no_train,
    test_steps,
):
    total_steps = int(float(total_steps))
    mt1_params = benchmark_evaluator.TrainingParameters(
        total_steps // 10, batch_size, None, architecture, False
    )
    ml1_params = benchmark_evaluator.TrainingParameters(
        total_steps // 10, batch_size, None, architecture, True
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
            test_steps,
            saved_model_name = "_".join(
                [
                    "sac",
                    "mt1",
                    "default",
                    "-".join([str(x) for x in architecture]),
                    test_class,
                    str(total_steps // 10),
                    str(batch_size),
                ]
            ),
            results_name="_".join(
                [
                    "sac",
                    "mt1",
                    "default",
                    "-".join([str(x) for x in architecture]),
                    test_class,
                    str(total_steps // 10),
                    str(test_steps),
                    str(batch_size),
                ]
            ),
        )
        if evaluate_ml_no_train:
            print("Evaluating ML1 with no training during testing")
            benchmark_evaluator.evaluate_benchmark(
                ml1,
                True,
                mt1_params,
                10,
                test_steps,
                saved_model_name="_".join(
                    [
                        "sac",
                        "metalearn1",
                        "default",
                        "-".join([str(x) for x in architecture]),
                        test_class,
                        str(total_steps // 10),
                        str(batch_size),
                    ]
                ),
                results_name="_".join(
                    [
                        "sac",
                        "metalearn1-notrain",
                        "default",
                        "-".join([str(x) for x in architecture]),
                        test_class,
                        str(total_steps // 10),
                        str(test_steps),
                        str(batch_size),
                    ]
                ),
            )
        print("Evaluating ML1 with training during testing")
        benchmark_evaluator.evaluate_benchmark(
            ml1,
            True,
            ml1_params,
            10,
            test_steps,
            saved_model_name=None,
            results_name="_".join(
                [
                    "sac",
                    "metalearn1",
                    "default",
                    "-".join([str(x) for x in architecture]),
                    test_class,
                    str(total_steps // 10),
                    str(test_steps),
                    str(batch_size),
                ]
            ),
        )
