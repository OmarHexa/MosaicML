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
    def get_registry(cls):
        return dict(cls.REGISTRY)

    @classmethod
    def get(cls, name):
        return cls.REGISTRY[name]
