import importlib
import os
import sys

def load_default_modules():
    importlib.import_module('minetorch.models')

def load_external_modules():
    pass
    # importlib.import_module('models')

class Model:
    def __init__(self, name, model_class):
        self.name = name
        self.model_class = model_class
        self.options = []

    def add_option(self, option):
        self.options.append(option)


class Option():

    def __init__(self, name, settings):
        self.name = name
        self.settings = settings


class Singleton():

    instance = None

    def __new__(cls):
        if not cls.instance:
            cls.instance = super().__new__(cls)
        return cls.instance


class ComponentDecorator(Singleton):

    # keep track of registed components
    registed_components = list()


    def register(self, component):
        if (component not in ComponentDecorator.registed_components):
            ComponentDecorator.registed_components.append(component)


class ModelDecorator(ComponentDecorator):
    """Decorate a model to be a Minetorch model.
    A minetorch model should be a higher order function or class which except some
    parameters and return a callable function, this returned function except one parameter
    which is the output value of the dataflow.
    """
    registed_models = list()

    def register(self, model):
        super().register(model)
        self.registed_models.append(model)

    def __call__(self, name):
        def inner_decorator(func):
            nonlocal name
            model = Model(name, func)
            for option in OptionDecorator.registed_options:
                model.add_option(option)
            OptionDecorator.registed_options = list()
            self.register(model)
            def __decorator(**kwargs):
                return func(**kwargs)
            return __decorator
        return inner_decorator


class OptionDecorator(Singleton):

    registed_options = list()

    def __call__(self, name, **settings):
        def inner_decorator(func):
            nonlocal name, settings
            self.registed_options.append(Option(name, settings))
            def __decorator(**kwargs):
                return func(**kwargs)
            return __decorator
        return inner_decorator


def optimizer():
    pass

def dataflow():
    pass

def loss():
    pass

class Choice():
    def __init__(self, collection):
        self.collection = collection

def boot():
    sys.path.insert(0, os.getcwd())
    load_default_modules()
    load_external_modules()
