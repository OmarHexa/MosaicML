from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Any
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.base import BaseEstimator
from automl.metrics.base import BaseMetricManager
from automl.metrics.classification import ClassificationMetricManager
from automl.tracker.base import BaseExperimentTracker
from ..hpo.factory import HPOFactory
from ..models import ModelFactory
@dataclass(order=True)
class OptimizedModelResult:
    sort_index: float = field(init=False, repr=False)
    score: float
    name: str
    best_params: dict
    model: Any
    metrics: dict

    def __post_init__(self):
        # For sorting: Higher score is better
        self.sort_index = -self.score
class BaseAutoML(ABC):
    """Base AutoML class"""
    def __init__(
        self,
        config: DictConfig,
        model_tracker: BaseExperimentTracker,
        logger: logging.Logger,
    ):
        self.cfg: OmegaConf = config
        self.hpo_config:dict = OmegaConf.to_container(self.cfg.hpo, resolve=True)
        self.models_config:dict = OmegaConf.to_container(self.cfg.models, resolve=True)
        self.model_tracker = model_tracker
        self.logger:logging.Logger = logger
        self.best_model: BaseEstimator = None
        self.best_model_name: str = None
        self.metric_manager: BaseMetricManager = self._init_metrics()
        self.optimized_models: list[OptimizedModelResult] = []

    def optimize_hyperparameters(self, model_name: str,model_info: dict, X, y) -> OptimizedModelResult:
        """Run hyperparameter optimization for a single model"""
        self.logger.info(f"Starting HPO for {model_name}...")
        model,param_space  = ModelFactory.get(model_info)
        # remove the strategy from hpo_config
        strategy = self.hpo_config.pop('strategy', 'randomizedsearch')
        optimizer = HPOFactory.get(
            name=strategy,
            estimator=model(),
            param_space=param_space,
            **self.hpo_config
        )
        
        # Run optimization
        optimizer.fit(X, y)
        
        # Get results
        best_model = optimizer.best_estimator_
        best_score = optimizer.best_score_
        best_params = optimizer.best_params_
        
        # Calculate metrics
        metrics = self.metric_manager(
            best_model, X, y
        )
        
        # Log to experiment tracker
        model_type = model.__module__.split('.')[0]  # Get library name
        self.model_tracker.log_model_run(
            model_name=model_name,
            params=best_params,
            metrics=metrics,
            model_type=model_type
        )
        
        return OptimizedModelResult(
        name=model_name,
        score=best_score,
        best_params=best_params,
        model=best_model,
        metrics=metrics,
        )

    @abstractmethod
    def _init_metrics(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        """Run full AutoML process"""
        pass

    def fit_best_model(self, X, y):
        """Train best model on full dataset"""
        if not self.best_model:
            raise ValueError("No best model selected. Run fit() first.")
        
        self.logger.info(f"Training final model ({self.best_model_name}) on full dataset...")
        self.best_model.fit(X, y)
        return self.best_model
    
    def evaluate(self, X, y) -> dict[str, float]:
        if not self.best_model:
            raise ValueError("No best model selected. Run fit() first.")
            
        return self.metric_manager(self.best_model, X, y)
    def get_leaderboard(self) -> pd.DataFrame:
        return pd.DataFrame([{
            'model': r.name,
            'score': r.score,
            'metrics': r.metrics,
            'params': r.best_params
        } for r in self.optimized_models])
    
class AutoClassifier(BaseAutoML):
    """Concrete AutoML implementation for classification"""
    
    def fit(self, X, y):
        # Run HPO for each model
        for name, model_info in self.models_config.items():
            try:
                result = self.optimize_hyperparameters(name,model_info, X, y)
                self.optimized_models.append(result)
                # Update best model
            except Exception as e:
                self.logger.error(f"HPO failed for {model_info[0]}: {str(e)}")
        self.optimized_models.sort()
        self.best_model = self.optimized_models[0].model
        self.best_model_name = self.optimized_models[0].name
        # Verify we have a best model
        if self.best_model is None:
            raise ValueError("No valid models were successfully trained")
        return self

    def _init_metrics(self):
        metric_config = self.cfg.metrics
        return ClassificationMetricManager(metric_config)