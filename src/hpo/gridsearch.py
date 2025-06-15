from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from hpo.base import BaseHPO, HPOFactory


@HPOFactory.register('gridsearch')
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