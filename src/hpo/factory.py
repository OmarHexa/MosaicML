from typing import Type

from sklearn.base import BaseEstimator

from .base import BaseHPO

class HPOFactory:
    """Factory class to create HPO adapters based on configuration"""
    registry = {}

    @classmethod
    def register(cls, name: str):
        """Register a new HPO adapter"""
        def decorator(adapter_class: Type[BaseHPO]):
            if not issubclass(adapter_class, BaseHPO):
                raise ValueError(f"{adapter_class.__name__} must inherit from BaseHPO")
            cls.registry[name] = adapter_class
            return adapter_class
        return decorator

    @staticmethod
    def get_optimizer(name: str, estimator: BaseEstimator, param_space: dict, **hpo_params) -> BaseHPO:
        """Get an instance of the specified HPO adapter"""
        name = name.lower()
        if name not in HPOFactory.registry:
            raise ValueError(f"HPO adapter '{name}' is not registered.")
        return HPOFactory.registry[name](estimator=estimator, param_space=param_space, **hpo_params)
