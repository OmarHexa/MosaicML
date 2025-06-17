from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from automl.core.registry import RegistryBase

class BaseHPO(ABC,metaclass=RegistryBase):
    """Abstract base class for HPO adapters"""
    __abstract__ = True

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
class BayesSearch(BaseHPO):
    """Adapter for BayesSearchCV"""
    def __init__(self, estimator: BaseEstimator, param_space: dict, **hpo_params):
        self.search = BayesSearchCV(
            estimator=estimator,
            search_spaces=param_space,
            **hpo_params
        )
    def fit(self, X, y):
        self.search.fit(X, y)
        return self
    @property
    def best_estimator_(self):
        return self.search.best_estimator_
    @property
    def best_score_(self):
        return self.search.best_score_
    @property
    def best_params_(self):
        return self.search.best_params_


class RandomizedSearch(BaseHPO):
    """Adapter for RandomizedSearchCV"""
    def __init__(self, estimator: BaseEstimator, param_space: dict, **hpo_params):
        self.search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_space,
            **hpo_params
        )
        
    def fit(self, X, y):
        self.search.fit(X, y)
        
    @property
    def best_estimator_(self) -> BaseEstimator:
        return self.search.best_estimator_
        
    @property
    def best_score_(self) -> float:
        return self.search.best_score_
        
    @property
    def best_params_(self) -> dict:
        return self.search.best_params_
    
class GridSearch(BaseHPO):
    """Adapter for GridSearchCV"""
    def __init__(self, estimator: BaseEstimator, param_space: dict, **hpo_params):
        self.search = GridSearchCV(
            estimator=estimator,
            param_grid=param_space,
            **hpo_params
        )
        
    def fit(self, X, y):
        self.search.fit(X, y)
        return self
    @property
    def best_estimator_(self):
        return self.search.best_estimator_
    @property
    def best_score_(self):
        return self.search.best_score_
    @property
    def best_params_(self):
        return self.search.best_params_
