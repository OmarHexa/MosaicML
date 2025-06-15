from sklearn.base import BaseEstimator
from skopt import BayesSearchCV
from .base import BaseHPO
from .factory import HPOFactory


@HPOFactory.register('bayessearch')
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



