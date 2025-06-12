from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from hydra.utils import instantiate
import optuna

class AutoClassifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.models_cfg = cfg.models
        self.preprocessing_cfg = cfg.preprocessing
        self.hpo_cfg = cfg.hpo

    def _build_preprocessing(self):
        steps = []
        if "imputer" in self.preprocessing_cfg:
            steps.append(("imputer", instantiate(self.preprocessing_cfg.imputer)))
        if "scaler" in self.preprocessing_cfg:
            steps.append(("scaler", instantiate(self.preprocessing_cfg.scaler)))
        return steps

    def _build_model(self, model_cfg, trial=None):
        params = {}
        for k, v in model_cfg.params.items():
            if isinstance(v, dict) and trial:
                if "min" in v and "max" in v:
                    params[k] = trial.suggest_int(k, v["min"], v["max"])
                elif "low" in v and "high" in v:
                    params[k] = trial.suggest_float(k, v["low"], v["high"])
            else:
                params[k] = v
        return instantiate(model_cfg.target, **params)

    def fit(self, X, y):
        preproc_steps = self._build_preprocessing()

        def objective(trial):
            best_score = 0
            for model_cfg in self.models_cfg:
                model = self._build_model(model_cfg, trial)
                pipeline = Pipeline(preproc_steps + [(model_cfg.name, model)])
                score = cross_val_score(pipeline, X, y, cv=3).mean()
                best_score = max(best_score, score)
            return best_score

        if self.hpo_cfg.enabled:
            study = optuna.create_study(direction=self.hpo_cfg.direction)
            study.optimize(objective, n_trials=self.hpo_cfg.n_trials)
            print("Best trial:", study.best_trial)
        else:
            for model_cfg in self.models_cfg:
                model = self._build_model(model_cfg)
                pipeline = Pipeline(preproc_steps + [(model_cfg.name, model)])
                score = cross_val_score(pipeline, X, y, cv=3).mean()
                print(f"{model_cfg.name}: {score:.4f}")
