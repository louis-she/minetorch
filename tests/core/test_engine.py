from ignite.engine.events import Events
from minetorch.core.plugin import Plugin, on, Event
from minetorch.core.engine import MineTorchEngine


class FooEvents(Event):
    EVENT_FOO = "event_foo"
    EVENT_BAR = "event_bar"


class FooPlugin(Plugin):

    def __init__(self, bar):
        self.bar = bar
        self.interval = 2

    @on(Events.EPOCH_STARTED | FooEvents.EVENT_FOO)
    def handler1(self):
        self.bar += 1

    @on(Events.ITERATION_COMPLETED, every=2)
    @on(FooEvents.EVENT_BAR)
    def handler2(self):
        self.bar += 2

    @on(Events.EPOCH_COMPLETED, lambda self: dict(every=self.interval))
    @on(FooEvents.EVENT_BAR)
    def handler3(self):
        self.bar += 3


FooPlugin.Events = FooEvents


def test_use_plugin():
    def forward_fn(a, b):
        pass

    engine = MineTorchEngine(forward_fn)
    plugin = FooPlugin(10)
    engine.use(plugin)

    assert len(engine._event_handlers) == 5
    assert plugin.bar == 10

    engine.fire_event(FooPlugin.Events.EVENT_FOO)
    assert plugin.bar == 11

    engine.fire_event(FooPlugin.Events.EVENT_BAR)
    assert plugin.bar == 16

    engine.fire_event(Events.EPOCH_STARTED)
    assert plugin.bar == 17


def test_event_attributes():
    def forward_fn(a, b):
        pass

    engine = MineTorchEngine(forward_fn)
    plugin = FooPlugin(10)
    engine.use(plugin)

    for i in range(5):
        engine.state.iteration += 1
        engine.fire_event(Events.ITERATION_COMPLETED)

    assert plugin.bar == 14

    for i in range(10):
        engine.state.epoch += 1
        engine.fire_event(Events.EPOCH_COMPLETED)

    assert plugin.bar == 29
