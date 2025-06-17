import logging
import numpy as np
from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


class ClassificationMetrics:
    """Metrics calculator for classification tasks"""
    def __init__(self, config: DictConfig):
        self.config = config
        self.objective = config.objective
        self.report = config.report
        self.metric_functions = {
            'accuracy': accuracy_score,
            'f1': f1_score,
            'roc_auc': roc_auc_score,
            'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')
        }

    def calculate_metrics(self, model: BaseEstimator, X, y) -> dict[str, float]:
        results = {}
        y_pred = model.predict(X)
        
        for metric in self.report:
            if metric in self.metric_functions:
                try:
                    if metric == 'roc_auc':
                        # ROC AUC requires probability scores
                        y_score = model.predict_proba(X)
                        if y_score.shape[1] == 2:  # Binary case
                            y_score = y_score[:, 1]
                        results[metric] = self.metric_functions[metric](y, y_score)
                    else:
                        results[metric] = self.metric_functions[metric](y, y_pred)
                except Exception as e:
                    logging.warning(f"Failed to calculate {metric}: {str(e)}")
                    results[metric] = np.nan
        return results
        
    def get_optimization_metric(self) -> str:
        return self.objective