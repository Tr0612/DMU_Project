import benchmark_trainer
import metaworld

benchmark_trainer.train_on_benchmark(
    metaworld.MT1("reach-v2"), "mt1", benchmark_trainer.TrainingParameters(2e4, 32)
)
