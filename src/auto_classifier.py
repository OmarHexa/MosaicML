import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import get_scorer
import numpy as np

from src.hpo_adapter import get_hpo_adapter

class AutoClassifier:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.scoring_metric = cfg.hpo.scoring
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
                'name': model_name
            })
        return models

    def fit(self, X, y):
        for model_info in self.models:
            print(f"\nTraining {model_info['name']} with {self.cfg.hpo._target_}...")
            
            # Get HPO adapter for this model
            hpo_adapter = get_hpo_adapter(
                cfg=self.cfg,
                estimator=model_info['instance'],
                param_space=model_info['param_space']
            )
            
            hpo_adapter.fit(X, y)
            
            if hpo_adapter.best_score_ > self.best_score:
                self.best_score = hpo_adapter.best_score_
                self.best_model = hpo_adapter.best_estimator_
                self.best_model_name = model_info['name']
                self.best_params = hpo_adapter.best_params_
        print(f"\nBest model: {self.best_model_name} with {self.scoring_metric}: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")

    def evaluate(self, X, y):
        scorer = get_scorer(self.scoring_metric)
        return scorer(self.best_model, X, y)

