# base.py
from abc import ABC, abstractmethod

import hydra
from omegaconf import DictConfig

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def score(self, X, y):
        # Optional, not required by all models
        return None


class ModelFactory:
    @staticmethod
    def get(model_cfg: DictConfig):
        target = hydra.utils.get_class(model_cfg.get("_target_"))
        params = model_cfg.get("params", {})
        return target, params