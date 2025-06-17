from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator

class BaseMetricManager(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.objective = config.objective
        self.report = config.report

    @abstractmethod
    def __call__(self, model: BaseEstimator, X, y) -> dict[str, float]:
        pass
    @property
    def objective(self) -> str:
        return self.objective