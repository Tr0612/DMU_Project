# import benchmark_evaluator
# import metaworld

# ml10 = metaworld.ML10()


# print("Evaluating ML")
# model, evaluation = benchmark_evaluator.evaluate_benchmark(
#     ml10,
#     True,
#     benchmark_evaluator.TrainingParameters(2e4, 32, None, [400, 400]),
#     evaluation_episodes=15,
#     checkpoint_frequency=None,
#     saved_model_name=None,
#     results_name="sac_metalearn10_default_2e4",
# )

import benchmark_evaluator
import metaworld


def run_experiment(
    total_steps,
    test_episodes,
    batch_size,
    architecture,
):
    total_steps = int(float(total_steps))
    ml10 = metaworld.ML10()
    ml10_params = benchmark_evaluator.TrainingParameters(
        total_steps, 32, None, [400, 400], True
    )
    benchmark_evaluator.evaluate_benchmark(
        ml10,
        True,
        ml10_params,
        test_episodes,
        saved_model_name=None,
        results_name="_".join(
            [
                "sac",
                "metalearn10",
                "default",
                "-".join([str(x) for x in architecture]),
                str(total_steps),
                str(batch_size),
            ]
        ),
    )
