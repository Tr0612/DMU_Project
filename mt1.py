import benchmark_trainer
import metaworld

benchmark_trainer.train_on_benchmark(
    metaworld.MT1("assembly-v2"), "mt1", benchmark_trainer.TrainingParameters(1e6)
)
