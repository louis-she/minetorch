from typing import Any, Callable

from ignite.engine.events import EventEnum, EventsList
from minetorch.core.statable import Statable
from minetorch.core.engine import MineTorchEngine


def on(event_name: Any, event_args_func: Callable = None, **event_args):
    if event_args_func and event_args:
        raise ValueError("Either event_args_func or key word argument")
    event_args = event_args_func if event_args_func is not None else event_args

    def decorator(func: Callable):
        if not hasattr(func, "__minetorch_events__"):
            func.__minetorch_events__ = []
        func.__minetorch_events__.append((event_name, event_args))
        return func
    return decorator


class Event(EventEnum):
    def __or__(self, other: Any) -> EventsList:
        return EventsList() | self | other


class Plugin(Statable):

    def init(self, engine: MineTorchEngine):
        self.engine = engine
