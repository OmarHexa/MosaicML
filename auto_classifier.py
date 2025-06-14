import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import get_scorer
import numpy as np

class AutoClassifier:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.models = self._init_models()
        self.best_model = None
        self.best_score = -np.inf
        
    def _init_models(self):
        models = []
            # Multiple models case
        for model_name, model_cfg in self.cfg.models.items():
            model_class = hydra.utils.get_class(model_cfg._target_)
            # Convert OmegaConf config to native Python types
            param_space = OmegaConf.to_container(model_cfg.param_space, resolve=True)
            models.append({
                'instance': model_class(),
                'param_space': param_space,
                'name': model_cfg._target_.split('.')[-1]
            })
        return models

    def fit(self, X, y):
        for model_info in self.models:
            print(f"\nTraining {model_info['name']}...")
            search = RandomizedSearchCV(
                estimator=model_info['instance'],
                param_distributions=model_info['param_space'],
                n_iter=self.cfg.base.n_iter,
                cv=self.cfg.base.cv_folds,
                scoring=self.cfg.base.metric,
                n_jobs=self.cfg.base.n_jobs,
                error_score='raise'
            )
            
            search.fit(X, y)
            
            if search.best_score_ > self.best_score:
                self.best_score = search.best_score_
                self.best_model = search.best_estimator_
                self.best_model_name = model_info['name']
                self.best_params = search.best_params_
                
        print(f"\nBest model: {self.best_model_name} with {self.cfg.base.metric}: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")

    def evaluate(self, X, y):
        scorer = get_scorer(self.cfg.base.metric)
        return scorer(self.best_model, X, y)

