from pathlib import Path


class FinalIntegrityCheck:

    def required_paths(self):

        return [
            "README.md",
            "requirements.txt",
            "football_quant_engine.spec",
            "database/schema.sql",
            "app/main.py",
            "app/controller.py",
            "engine/job_runner.py",
            "engine/offline_engine.py",
            "engine/plugin_loader.py",
            "engine/prediction_pipeline.py",
            "analysis/probability_markets.py",
            "analysis/report_generator.py",
            "features/feature_engine.py",
            "features/offline_features.py",
            "models/poisson_model.py",
            "models/bivariate_poisson.py",
            "models/elo_model.py",
            "models/bayesian_model.py",
            "training/dataset_builder.py",
            "training/backtest_engine.py",
            "training/backtest_metrics.py",
            "ui/main_window.py",
        ]

    def missing_paths(self):

        missing = []

        for item in self.required_paths():
            if not Path(item).exists():
                missing.append(item)

        return missing

    def run(self):

        missing = self.missing_paths()

        return {
            "ok": len(missing) == 0,
            "missing": missing,
            "checked_count": len(self.required_paths()),
        }
