from abc import ABC, ABCMeta, abstractmethod
from sklearn.base import BaseEstimator


class RegistryBase(ABCMeta):
    REGISTRY = {}

    def __new__(mcs, name, bases,namespace, **kwargs):
        new_cls = super().__new__(mcs, name, bases,namespace, **kwargs)
        reg_name = new_cls.__name__.lower()
        if reg_name not in mcs.REGISTRY and "base" not in name.lower():
            mcs.REGISTRY[reg_name] = new_cls
        return new_cls
    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)

    @classmethod
    def get(cls, name):
        return cls.REGISTRY[name]
    
class BaseHPO(ABC,metaclass=RegistryBase):
    """Abstract base class for HPO adapters"""

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