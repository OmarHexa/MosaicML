from abc import ABC, abstractmethod


class BaseExperimentTracker(ABC):
    """Abstract base class for experiment tracking"""
    @abstractmethod
    def log_model_run(self, model_name: str, params: dict, metrics: dict, model_type: str):
        """Log a model run with parameters and metrics"""
        pass
        
    @abstractmethod
    def log_final_model(self, model_name: str, params: dict, metrics: dict):
        """Log the final selected model"""
        pass
        
    @abstractmethod
    def log_artifact(self, artifact_path: str, artifact_name: str):
        """Log an artifact file"""
        pass