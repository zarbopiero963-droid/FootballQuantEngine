from training.backtest_engine import BacktestEngine
from training.dataset_builder import DatasetBuilder


class TrainingManager:

    def __init__(self):

        self.dataset_builder = DatasetBuilder()
        self.backtest_engine = BacktestEngine()

    def build_dataset(self):

        return self.dataset_builder.build_training_dataset()

    def run_backtest(self):

        return self.backtest_engine.run()
