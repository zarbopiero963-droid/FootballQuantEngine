import importlib.util



class LocalAutoML:

    def _has_lib(self, name):

        return importlib.util.find_spec(name) is not None

    def available_models(self):

        models = []

        if self._has_lib("sklearn"):
            models.append("sklearn_random_forest")

        if self._has_lib("xgboost"):
            models.append("xgboost")

        if self._has_lib("lightgbm"):
            models.append("lightgbm")

        if self._has_lib("catboost"):
            models.append("catboost")

        if self._has_lib("optuna"):
            models.append("optuna")

        return models

    def run(self, dataset_df):

        if dataset_df.empty:
            return {
                "available_models": self.available_models(),
                "best_model": None,
                "status": "empty_dataset",
            }

        if "target_home_win" not in dataset_df.columns:
            return {
                "available_models": self.available_models(),
                "best_model": None,
                "status": "missing_target",
            }

        available_models = self.available_models()

        return {
            "available_models": available_models,
            "best_model": available_models[0] if available_models else None,
            "rows": len(dataset_df),
            "status": "ready",
        }
