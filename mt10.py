import benchmark_trainer
import metaworld

benchmark_trainer.train_on_benchmark(
    metaworld.MT10(), "mt10", benchmark_trainer.TrainingParameters(2e5, 32)
)
