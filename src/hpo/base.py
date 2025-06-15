from abc import ABC, abstractmethod
from typing import Type
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

class HPOFactory:
    """Factory class to create HPO adapters based on configuration"""
    registry = {}

    @classmethod
    def register(cls, name: str):
        """Register a new HPO adapter"""
        def decorator(adapter_class: Type[BaseHPO]):
            if not issubclass(adapter_class, BaseHPO):
                raise ValueError(f"{adapter_class.__name__} must inherit from BaseHPO")
            cls.registry[name] = adapter_class
            return adapter_class
        return decorator

    @staticmethod
    def get_optimizer(name: str, estimator: BaseEstimator, param_space: dict, **hpo_params) -> BaseHPO:
        """Get an instance of the specified HPO adapter"""
        if name not in HPOFactory.registry:
            raise ValueError(f"HPO adapter '{name}' is not registered.")
        return HPOFactory.registry[name](estimator, param_space, **hpo_params)
    
