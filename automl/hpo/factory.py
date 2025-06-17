from abc import ABCMeta


class RegistryBase(ABCMeta):
    REGISTRY = {}

    def __new__(mcs, name, bases, namespace, **kwargs):
        new_cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        name = name.lower()
        if name not in mcs.REGISTRY and not namespace.get("__abstract__",False):
            mcs.REGISTRY[name] = new_cls
        return new_cls
    
    @classmethod
    def list(cls):
        return dict(cls.REGISTRY)

    @classmethod
    def get(cls, name):
        return cls.REGISTRY[name]

class HPOFactory:
    @staticmethod
    def get(name, estimator, param_space, **kwargs):
        name = name.lower()
        registry = RegistryBase.list()
        if name not in registry:
            raise ValueError(f"HPO adapter '{name}' not registered.")
        return registry[name](estimator, param_space, **kwargs)

    @staticmethod
    def list_available():
        return list(RegistryBase.get_registry().keys())