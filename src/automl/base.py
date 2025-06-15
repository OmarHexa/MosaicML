# automl/base.py
from abc import ABC, abstractmethod
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator
from typing import List, Tuple, Optional
from src.hpo.factory import HPOFactory
from src.models.base import BaseModelInitializer, SklearnModelInitializer
from src.tracker.base import BaseExperimentTracker
from src.metrics.base import BaseMetricsCalculator

class BaseAutoML(ABC):
    """Base AutoML class"""
    def __init__(
        self,
        config: DictConfig,
        experiment_tracker: BaseExperimentTracker,
        metrics_calculator: BaseMetricsCalculator,
        logger: logging.Logger,
        model_initializer: BaseModelInitializer = SklearnModelInitializer(),
    ):
        self.config = config
        self.model_initializer = model_initializer
        self.experiment_tracker = experiment_tracker
        self.metrics_calculator = metrics_calculator
        self.logger = logger
        
        self.models = []
        self.best_model: Optional[BaseEstimator] = None
        self.best_model_name: Optional[str] = None
        self.best_params: Optional[dict] = None
        self.best_score: float = -np.inf
        self.model_rankings: List[dict] = []

    @property
    def model_list(self) -> List[str]:
        """Return names of all models to be tested"""
        return [name for name, _, _ in self.models]

    def initialize(self):
        """Initialize all models"""
        self.logger.info("Initializing models...")
        self.models = self.model_initializer.initialize_models(self.config.models)
        self.logger.info(f"Models initialized: {self.model_list}")

    def _run_hpo(self, model_info: Tuple[str, BaseEstimator, dict], X, y):
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
        metrics = self.metrics_calculator.calculate_metrics(
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
            "metrics": metrics
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
    def generate_report(self) -> str:
        """Generate model performance report"""
        pass

    def train_final_model(self, X, y):
        """Train best model on full dataset"""
        if not self.best_model:
            raise ValueError("No best model selected. Run fit() first.")
        
        self.logger.info(f"Training final model ({self.best_model_name}) on full dataset...")
        self.best_model.fit(X, y)
        return self.best_model