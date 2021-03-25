from __future__ import annotations
import secrets
from pathlib import Path
from typing import Callable

from ignite.engine.events import EventsList

from minetorch.core.statable import Statable
from ignite.engine import Engine

__all__ = ["MineTorchEngine"]


class MineTorchEngine(Engine, Statable):
    """A engine that support plugin register
    """

    def __init__(self, process_function: Callable, root: str = None, code: str = None):
        super().__init__(process_function)
        self.code = secrets.token_hex(6) if code is None else code
        self.root = (Path().home() / ".minetorch_experiments").as_posix() if root is None else root
        self.exp_dir = Path(self.root) / self.code

    def use(self, plugin):
        """Use a plugin
        """
        if hasattr(plugin, "Events"):
            self.register_events(*plugin.Events)

        for attr_name in dir(plugin):
            handler = getattr(plugin, attr_name)
            if not (callable(handler) and hasattr(handler, "__minetorch_events__")):
                continue
            for event_name, event_args in handler.__minetorch_events__:
                event_args = event_args(plugin) if callable(event_args) else event_args
                if isinstance(event_name, EventsList) or not event_args:
                    self.add_event_handler(event_name, handler)
                else:
                    self.add_event_handler(event_name(**event_args), handler)
        plugin.init(self)
