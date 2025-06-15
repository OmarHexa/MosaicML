import importlib
import pkgutil

def autodiscover_adapters():
    for _, module_name, _ in pkgutil.iter_modules(__path__):
        importlib.import_module(f"src.hpo.{module_name}")

# Call autodiscover immediately
autodiscover_adapters()