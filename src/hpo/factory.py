from .base import RegistryBase

class HPOFactory:
    @staticmethod
    def get_optimizer(name, estimator, param_space, **kwargs):
        name = name.lower()
        registry = RegistryBase.get_registry()
        if name not in registry:
            raise ValueError(f"HPO adapter '{name}' not registered.")
        return registry[name](estimator, param_space, **kwargs)

    @staticmethod
    def list_available():
        return list(RegistryBase.get_registry().keys())