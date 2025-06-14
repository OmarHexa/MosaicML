from abc import ABC, abstractmethod
from omegaconf import DictConfig
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV
# from skopt import BayesSearchCV  # Requires scikit-optimize

class BaseHPOAdapter(ABC):
    """Base class for HPO strategy adapters"""
    def __init__(self, estimator, param_space, **kwargs):
        self.estimator = estimator
        self.param_space = param_space
        self.kwargs = kwargs
        self._init_search()
        
    @abstractmethod
    def _init_search(self):
        """Initialize the HPO search strategy"""
        pass
        
    @abstractmethod
    def fit(self, X, y):
        """Execute the hyperparameter search"""
        pass
        
    @property
    @abstractmethod
    def best_estimator_(self):
        """Return the best estimator found"""
        pass
        
    @property
    @abstractmethod
    def best_score_(self):
        """Return the best score found"""
        pass
        
    @property
    @abstractmethod
    def best_params_(self):
        """Return the best parameters found"""
        pass


class RandomizedSearchAdapter(BaseHPOAdapter):
    """Adapter for RandomizedSearchCV"""
    def _init_search(self):
        self.search = RandomizedSearchCV(
            estimator=self.estimator,
            param_distributions=self.param_space,
            **self.kwargs
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

class GridSearchAdapter(BaseHPOAdapter):
    """Adapter for GridSearchCV"""
    def _init_search(self):
        self.search = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_space,
            **self.kwargs
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

class BayesSearchAdapter(BaseHPOAdapter):
    """Adapter for BayesSearchCV"""
    def _init_search(self):
        self.search = BayesSearchCV(
            estimator=self.estimator,
            search_spaces=self.param_space,
            **self.kwargs
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
    
def get_hpo_adapter(cfg: DictConfig, estimator, param_space) -> BaseHPOAdapter:
    """Factory for HPO adapters based on configuration"""
    strategy = cfg.hpo._target_.split('.')[-1].lower()
    
    # Extract common HPO parameters
    common_params = {
        'cv': cfg.hpo.get('cv', 5),
        'scoring': cfg.hpo.get('scoring', 'accuracy'),
        'n_jobs': cfg.hpo.get('n_jobs', -1),
        'verbose': cfg.hpo.get('verbose', 1),
        'error_score': cfg.hpo.get('error_score', 'raise'),
    }
    
    # Add strategy-specific parameters
    if strategy == 'randomizedsearchcv':
        common_params.update({
            'n_iter': cfg.hpo.n_iter,
            'random_state': cfg.hpo.get('random_state', None)
        })
        return RandomizedSearchAdapter(estimator, param_space, **common_params)
        
    elif strategy == 'gridsearchcv':
        return GridSearchAdapter(estimator, param_space, **common_params)
    elif strategy == 'bayessearchcv':
        common_params.update({
            'n_iter': cfg.hpo.n_iter,
            'random_state': cfg.hpo.get('random_state', None)
        })
        return BayesSearchAdapter(estimator, param_space, **common_params)
        
    else:
        raise ValueError(f"Unsupported HPO strategy: {strategy}")