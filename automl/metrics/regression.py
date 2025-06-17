
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    root_mean_squared_error
)
from .base import BaseMetricManager
import numpy as np
import logging


class RegressionMetricManager(BaseMetricManager):
    """Handles standard regression metrics"""

    def __init__(self, config):
        super().__init__(config)

        self.metric_functions = {
            'mse': mean_squared_error,
            'rmse': lambda y, p: root_mean_squared_error(y, p, squared=False),
            'mae': mean_absolute_error,
            'mape': mean_absolute_percentage_error,
            'r2': r2_score,
        }

    def __call__(self, model, X, y) -> dict[str, float]:
        y_pred = model.predict(X)
        results = {}

        for metric in self.report:
            if metric not in self.metric_functions:
                logging.warning(f"Unknown metric: {metric}")
                continue
            try:
                results[metric] = self.metric_functions[metric](y, y_pred)
            except Exception as e:
                logging.warning(f"Failed to calculate {metric}: {e}")
                results[metric] = np.nan
        return results
