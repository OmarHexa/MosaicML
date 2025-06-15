from abc import ABC, abstractmethod
import logging
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator

class BaseModelInitializer(ABC):
    """Abstract base class for model initialization"""
    @abstractmethod
    def initialize_models(self, model_configs: DictConfig) -> list[tuple[str, BaseEstimator, dict]]:
        """
        Initialize models from configuration
        Returns list of (model_name, model_instance, param_space)
        """
        pass

    @abstractmethod
    def validate_model(self, model: BaseEstimator):
        """Validate model interface meets requirements"""
        pass

class SklearnModelInitializer(BaseModelInitializer):
    """Model initializer for scikit-learn compatible models"""
    def initialize_models(self, model_configs: DictConfig) -> list[tuple[str, BaseEstimator, dict]]:
        models = []
        for model_name, model_cfg in model_configs.items():
            try:
                model_class = get_class(model_cfg._target_)
                # Validate model interface
                self.validate_model(model_class)
                param_space = OmegaConf.to_container(model_cfg.param_space, resolve=True)
                models.append((model_name, model_class, param_space))
            except Exception as e:
                logging.error(f"Error initializing {model_name}: {str(e)}")
        return models

    def validate_model(self, model: BaseEstimator):
        required_methods = ['fit', 'predict', 'set_params', 'get_params']
        if hasattr(model, 'predict_proba'):
            required_methods.append('predict_proba')
            
        for method in required_methods:
            if not hasattr(model, method):
                raise ValueError(f"Model {type(model).__name__} missing required method: {method}")
