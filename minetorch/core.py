import importlib
import os
import sys

registed_models = []

def load_default_modules():
    importlib.import_module('minetorch.models')

def load_external_modules():
    pass
    # importlib.import_module('models')

def model(func):
    """Decorate a model to be a Minetorch model.
    A minetorch model should be a higher order function or class which except some
    parameters and return a callable function, this returned function except one parameter
    which is the output value of the dataflow.
    """
    global registed_models
    registed_models.append(func)
    def __decorator():
        return func()
    return __decorator

def optimizer():
    pass

def dataflow():
    pass

def loss():
    pass

class Choice():
    def __init__(self, collection):
        self.collection = collection

def option(name, **kwargs):
    def inner_decorator(func):
        def __decorator():
            return func(name, kwargs)
        return __decorator
    return inner_decorator

def boot():
    sys.path.insert(0, os.getcwd())
    load_default_modules()
    load_external_modules()
