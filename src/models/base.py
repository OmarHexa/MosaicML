from abc import ABC, abstractmethod

from omegaconf import DictConfig
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
