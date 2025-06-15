from sklearn.base import clone
from .base import BaseAutoML

class AutoClassifier(BaseAutoML):
    """Concrete AutoML implementation for classification"""
    def fit(self, X, y):
        self.initialize()
        self.model_rankings = []
        
        # Run HPO for each model
        for model_info in self.models:
            try:
                result = self._run_hpo(model_info, X, y)
                self.model_rankings.append(result)
                
                # Update best model
                opt_metric = self.metrics_calculator.get_optimization_metric()
                score = result['metrics'][opt_metric]
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = clone(result['model'])
                    self.best_model_name = result['model_name']
                    self.best_params = result['params']
            except Exception as e:
                self.logger.error(f"HPO failed for {model_info[0]}: {str(e)}")
        
        # Log best model
        metrics = self.metrics_calculator.calculate_metrics(
            self.best_model, X, y, self.config.metrics
        )
        self.experiment_tracker.log_final_model(
            model_name=self.best_model_name,
            params=self.best_params,
            metrics=metrics
        )
        
        # Sort rankings by optimization metric
        opt_metric = self.metrics_calculator.get_optimization_metric()
        self.model_rankings.sort(key=lambda x: x['metrics'][opt_metric], reverse=True)
        
        return self

    def evaluate(self, X, y) -> dict[str, float]:
        if not self.best_model:
            raise ValueError("No best model selected. Run fit() first.")
            
        return self.metrics_calculator.calculate_metrics(
            self.best_model, X, y, self.config.metrics
        )

    def generate_report(self) -> str:
        if not self.best_model:
            raise ValueError("No best model selected. Run fit() first.")
            
        report = f"AutoML Classification Report\n{'='*30}\n"
        report += f"Best Model: {self.best_model_name}\n"
        report += f"Optimization Metric: {self.metrics_calculator.get_optimization_metric()}\n"
        report += f"Best Score: {self.best_score:.4f}\n\n"
        
        report += "Model Rankings:\n"
        for i, rank in enumerate(self.model_rankings):
            report += f"{i+1}. {rank['model_name']}: {rank['score']:.4f}\n"
        
        return report