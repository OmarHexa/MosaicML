from omegaconf import DictConfig
from tracker.base import BaseExperimentTracker


class MLflowExperimentTracker(BaseExperimentTracker):
    """Experiment tracker using MLflow"""
    def __init__(self, config: DictConfig):
        import mlflow
        self.mlflow = mlflow
        self.config = config
        self._setup()
        
    def _setup(self):
        self.mlflow.set_tracking_uri(self.config.tracking_uri)
        self.mlflow.set_experiment(self.config.experiment_name)
        self.mlflow.start_run()
        
    def log_model_run(self, model_name: str, params: dict, metrics: dict, model_type: str):
        with self.mlflow.start_run(nested=True, run_name=model_name):
            self.mlflow.log_params(params)
            self.mlflow.log_metrics(metrics)
            self.mlflow.set_tag("model_type", model_type)

    def log_final_model(self, model_name: str, params: dict, metrics: dict):
        self.mlflow.log_params(params)
        self.mlflow.log_metrics(metrics)
        self.mlflow.set_tag("best_model", model_name)
        
    def log_artifact(self, artifact_path: str, artifact_name: str):
        self.mlflow.log_artifact(artifact_path, artifact_name)