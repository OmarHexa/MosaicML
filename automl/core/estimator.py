from automl.metrics.classification import ClassificationMetrics
from abc import ABC, abstractmethod
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator
from typing import Any, List, Tuple, Optional
from automl.models.sklearn import BaseModelInitializer, SklearnModelInitializer
from automl.tracker.base import BaseExperimentTracker
from .factory import HPOFactory

class BaseAutoML(ABC):
    """Base AutoML class"""
    def __init__(
        self,
        config: DictConfig,
        experiment_tracker: BaseExperimentTracker,
        logger: logging.Logger,
        model_initializer: BaseModelInitializer = SklearnModelInitializer(),
    ):
        self.config = config
        self.model_initializer = model_initializer
        self.experiment_tracker = experiment_tracker
        self.logger = logger
        
        self.models = []
        self.best_model: Optional[BaseEstimator] = None
        self.best_model_name: Optional[str] = None
        self.best_params: Optional[dict] = None
        self.best_score: float = -np.inf
        self.model_rankings: List[dict] = []

        self.metrics_reporter = self._set_metrics_reporter()
        self.initialize()

    @property
    def model_list(self) -> List[str]:
        """Return names of all models to be tested"""
        return [name for name, _, _ in self.models]

    def initialize(self):
        """Initialize all models"""
        self.logger.info("Initializing models...")
        self.models = self.model_initializer.initialize_models(self.config.models)
        self.logger.info(f"Models initialized: {self.model_list}")

    def search_hyperparameter(self, model_info: Tuple[str, BaseEstimator, dict], X, y) -> dict:
        """Run hyperparameter optimization for a single model"""
        model_name, model, param_space = model_info
        self.logger.info(f"Starting HPO for {model_name}...")
        
        # Create HPO adapter
        hpo_config = OmegaConf.to_container(self.config.hpo, resolve=True)

        strategy = hpo_config.pop('strategy', 'randomizedsearch')
        optimizer = HPOFactory.get_optimizer(
            name=strategy,
            estimator=model(),
            param_space=param_space,
            **hpo_config
        )
        
        # Run optimization
        optimizer.fit(X, y)
        
        # Get results
        best_model = optimizer.best_estimator_
        best_score = optimizer.best_score_
        best_params = optimizer.best_params_
        
        # Calculate metrics
        metrics = self.metrics_reporter.calculate_metrics(
            best_model, X, y
        )
        
        # Log to experiment tracker
        model_type = model.__module__.split('.')[0]  # Get library name
        self.experiment_tracker.log_model_run(
            model_name=model_name,
            params=best_params,
            metrics=metrics,
            model_type=model_type
        )
        
        return {
            "model_name": model_name,
            "model": best_model,
            "params": best_params,
            "score": best_score,
            "report": metrics
        }

    @abstractmethod
    def fit(self, X, y):
        """Run full AutoML process"""
        pass

    @abstractmethod
    def evaluate(self, X, y) -> dict[str, float]:
        """Evaluate best model on new data"""
        pass

    @abstractmethod
    def _set_metrics_reporter(self) -> Any:
        """Set the metrics reporter"""
        pass

    def train_final_model(self, X, y):
        """Train best model on full dataset"""
        if not self.best_model:
            raise ValueError("No best model selected. Run fit() first.")
        
        self.logger.info(f"Training final model ({self.best_model_name}) on full dataset...")
        self.best_model.fit(X, y)
        return self.best_model
class AutoClassifier(BaseAutoML):
    """Concrete AutoML implementation for classification"""
    def _set_metrics_reporter(self):
        return ClassificationMetrics(self.config.metrics)
    
    def fit(self, X, y):
        # Run HPO for each model
        for model_info in self.models:
            try:
                result = self.search_hyperparameter(model_info, X, y)
                self.model_rankings.append(result)
                # Update best model
                opt_metric = self.metrics_reporter.get_optimization_metric()
                score = result['report'][opt_metric]     
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = result['model']
                    self.best_model_name = result['model_name']
                    self.best_params = result['params']
            except Exception as e:
                self.logger.error(f"HPO failed for {model_info[0]}: {str(e)}")
        # Verify we have a best model
        if self.best_model is None:
            raise ValueError("No valid models were successfully trained")
        
        best_result = next(
        r for r in self.model_rankings 
        if r['model_name'] == self.best_model_name
    )
        self.experiment_tracker.log_final_model(
        model_name=self.best_model_name,
        params=self.best_params,
        metrics=best_result['report']
    )
        
        # Sort rankings by optimization metric
        opt_metric = self.metrics_reporter.get_optimization_metric()
        self.model_rankings.sort(key=lambda x: x['report'][opt_metric], reverse=True)
        
        return self

    def evaluate(self, X, y) -> dict[str, float]:
        if not self.best_model:
            raise ValueError("No best model selected. Run fit() first.")
            
        return self.metrics_reporter.calculate_metrics(
            self.best_model, X, y
        )