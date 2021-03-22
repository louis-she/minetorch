from ignite.engine import Engine
from minetorch.core.plugin import Plugin
from typing import Callable

__all__ = ["MineTorchEngine"]

class MineTorchEngine(Engine):
    """A engine that support plugin register
    """

    def __init__(self, process_function: Callable):
        super().__init__(process_function)

    def use(self, plugin: Plugin):
        """Use a plugin
        """
