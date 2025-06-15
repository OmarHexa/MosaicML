from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from hpo.base import BaseHPO, HPOFactory


@HPOFactory.register('randomsearch')
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
    


