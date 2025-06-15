
from src.metrics.classification import ClassificationMetrics
from .base import BaseAutoML

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