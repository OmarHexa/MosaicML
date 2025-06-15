from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator



class BaseHPO(ABC):
    """Abstract base class for HPO adapters"""
    @abstractmethod
    def __init__(self, estimator: BaseEstimator, param_space: dict, **hpo_params):
        if not hasattr(estimator, 'fit') or not callable(getattr(estimator, 'fit')):
            raise ValueError("Estimator must have a 'fit' method")
        if not hasattr(estimator, 'predict') or not callable(getattr(estimator, 'predict')):
            raise ValueError("Estimator must have a 'predict' method")
        pass
        
    @abstractmethod
    def fit(self, X, y):
        """Execute hyperparameter optimization"""
        pass
        
    @property
    @abstractmethod
    def best_estimator_(self) -> BaseEstimator:
        pass
        
    @property
    @abstractmethod
    def best_score_(self) -> float:
        pass
        
    @property
    @abstractmethod
    def best_params_(self) -> dict:
        pass

