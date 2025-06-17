from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, log_loss
)
from .base import BaseMetricManager
import numpy as np
import logging

class ClassificationMetricManager(BaseMetricManager):
    """Handles binary, multiclass, and multilabel metrics"""

    def __init__(self, config, task_type='binary'):
        super().__init__(config)
        self.task_type = task_type

        self.metric_functions = {
            'accuracy': accuracy_score,
            'f1': lambda y, p: f1_score(y, p, average='binary'),
            'f1_macro': lambda y, p: f1_score(y, p, average='macro'),
            'precision': lambda y, p: precision_score(y, p, average='binary'),
            'recall': lambda y, p: recall_score(y, p, average='binary'),
            'roc_auc': self._roc_auc,
            'log_loss': log_loss
        }

    def _roc_auc(self, y_true, model, X):
        try:
            y_score = model.predict_proba(X)
            if y_score.shape[1] == 2:
                y_score = y_score[:, 1]
            return roc_auc_score(y_true, y_score)
        except Exception as e:
            logging.warning(f"Failed roc_auc: {e}")
            return np.nan

    def __call__(self, model, X, y) -> dict[str, float]:
        y_pred = model.predict(X)
        results = {}
        for metric in self.report:
            if metric not in self.metric_functions:
                logging.warning(f"Unknown metric: {metric}")
                continue
            try:
                if metric == 'roc_auc':
                    results[metric] = self.metric_functions[metric](y, model, X)
                else:
                    results[metric] = self.metric_functions[metric](y, y_pred)
            except Exception as e:
                logging.warning(f"Failed {metric}: {e}")
                results[metric] = np.nan
        return results
