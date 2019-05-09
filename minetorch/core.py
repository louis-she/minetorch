import importlib
import os
import sys
import json

def load_default_modules():
    importlib.import_module('minetorch.datasets')
    importlib.import_module('minetorch.dataflows')
    importlib.import_module('minetorch.models')
    importlib.import_module('minetorch.optimizers')
    importlib.import_module('minetorch.losses')

def load_external_modules():
    pass
    # importlib.import_module('models')


class Component():

    def __init__(self, name, component_class):
        self.name = name
        self.component_class = component_class
        self.options = []

    def add_option(self, option):
        self.options.append(option)

    def to_json_serializable(self):
        return {
            'name': self.name,
            'options': list(map(lambda o: o.to_json_serializable(), self.options))
        }


class Model(Component):
    pass

class Dataset(Component):
    pass

class Dataflow(Component):
    pass

class Optimizer(Component):
    pass

class Loss(Component):
    pass


class Option():

    def __init__(self, name, settings):
        self.name = name
        self.settings = settings

    def to_json_serializable(self):
        return {
            'name': self.name,
            'settings': self.settings
        }


class Singleton():

    instance = None

    def __new__(cls):
        if not cls.instance:
            cls.instance = super().__new__(cls)
        return cls.instance


class ComponentDecorator(Singleton):

    # keep track of registed components
    registed_components = list()

    # should be override in subclass
    component_class = Component

    def __call__(self, name):
        def inner_decorator(func):
            nonlocal name
            model = self.component_class(name, func)
            for option in OptionDecorator.registed_options:
                model.add_option(option)
            OptionDecorator.registed_options = list()
            self.register(model)
            def __decorator(**kwargs):
                return func(**kwargs)
            return __decorator
        return inner_decorator


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
    component_class = Model

    def register(self, model):
        super().register(model)
        self.registed_models.append(model)


class DataflowDecorator(ComponentDecorator):
    """Decorate a ProcessorBundler to be a Minetorch Dataflow.
    Minetorch Dataflow is Minetorch's ProcessorBundler, this decorator is just a wrap for
    ProcessorBundler.
    """
    registed_dataflows = list()
    component_class = Dataflow

    def register(self, dataflow):
        super().register(dataflow)
        self.registed_dataflows.append(dataflow)


class DatasetDecorator(ComponentDecorator):
    """Decorate a dataset to be a Minetorch model.
    In Minetorch, dataset can be anything that can be indexed(should implement [] operation)
    and have a length(should implement __len__ function), just like `torch.utils.data.Dataset`.

    The Dataset is a table(csv) like annotation which includes all the necessary information
    about your dataset. And let the `Dataflow` to deal with the transforms. In this way Minetorch
    could track and even visuzalize how your data changes in the dataflow from the start to end. But
    it's also OK to hanle some base logic in your Dataset definition.

    Here's a example of image classification scenario:

    Recommended data flow:
        1. create a Dataset() which yield ($path_to_image, $scalar_class_of_image)
        2. create a Dataflow() named `ImageLoader` which will transfer the first column of
           the data from $path_to_image to $tensor_of_image
        3. create a Dataflow() named `ImageResizer` which will transfer the first column of
           the data from $tensor_of_image to $fixed_size_tensor_of_image
        4. create a Dataflow() named `OneHotEncode` which will transfer the second column of
           the data from $scalar_class_of_image to $vector_class_of_image

    But you can always do all of the 4 steps in the Datasets:
        1. create a Dataset() which yield ($fixed_size_tensor_of_image, $vector_class_of_image)
    This is alright but what it doese is mixed the dataset(which is a information) and
    dataflow(which is actions) in one place.

    It's always recommanded to use the previous one since it will take more advantages of the
    power of Minetorch's web dashboard.
    """
    registed_datasets = list()
    component_class = Dataset

    def register(self, dataset):
        super().register(dataset)
        self.registed_datasets.append(dataset)


class OptimizerDecorator(ComponentDecorator):
    """Run backpropagation for model
    TODO: currenty this is just a wrap PyTorch optimizer, if want to support more
    platform, should refactor this
    """
    registed_optimizers = list()
    component_class = Optimizer

    def register(self, optimizer):
        super().register(optimizer)
        self.registed_optimizers.append(optimizer)


class LossDecorator(ComponentDecorator):
    """Make a function to a Minetorch Loss object
    The function to be decorated must received 2 params, first one is yield by
    dataflow and the second one is the output of the model.
    The return value of the function must be a scalar.
    """
    registed_losses = list()
    component_class = Loss

    def register(self, loss):
        super().register(loss)
        self.registed_losses.append(loss)


class OptionDecorator(Singleton):
    """Add an option to any component.
    A component often come with many options, like `fold` for a Dataset component,
    `learning rate` for Optimizer component. To let user tweak these parameters
    throught the web interface, Minetorch extender should describe what kind option a
    component can received through this decorator.

    The @option decorator should always be called **after** component decorator, see
    Minetorch built in components for examples.
    """

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
