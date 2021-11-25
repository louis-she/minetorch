from minetorch.miner import Miner
from minetorch.plugin import Plugin


class PluginWithState(Plugin):

    __state_members__ = ['state_1', 'state_2']

    def __init__(self, state_1=None, state_2=None, prefix=""):
        super().__init__(prefix)
        self.state_1 = state_1
        self.state_2 = state_2


class PluginWithTriggeredEvent(Plugin):

    __events__ = ["SOME_EVENT", "SOME_OTHER_EVENT"]

    def after_epoch_end(self):
        super().after_epoch_end()
        self.trigger("SOME_EVENT", {"key": "value"})
        self.trigger("SOME_OTHER_EVENT", {"other_key": "other_value"})


class PluginWithBeforeHandler(Plugin):

    def __init__(self, prefix=""):
        super().__init__(prefix)
        self.allowed_hooks = [
            'before_epoch_start',
            'after_epoch_end',
        ]

        self.called_times = {
            'before_epoch_start': 0,
            'after_epoch_end': 0,
            'after_train_iteration_end': 0
        }

    def before_handler(self, hook_point: str, payload):
        return hook_point in self.allowed_hooks

    def before_epoch_start(self):
        self.called_times['before_epoch_start'] += 1

    def after_epoch_end(self):
        self.called_times['after_epoch_end'] += 1

    def after_train_iteration_end(self, **ignore):
        self.called_times['after_train_iteration_end'] += 1


def test_plugin_persist(miner_factory):
    plugin = PluginWithState(state_1='Some string', state_2={'key': 'value'})
    miner : Miner = miner_factory(plugins=[plugin])
    miner.persist('test')

    plugin_resume = PluginWithState(state_1='Some other string')
    assert plugin_resume.state_1 == 'Some other string'
    assert plugin_resume.state_2 is None

    miner_factory(plugins=[plugin_resume], resume="test")
    assert plugin_resume.state_1 == 'Some string'
    assert plugin_resume.state_2['key'] == 'value'


def test_plugin_before_handler(miner_factory):
    plugin = PluginWithBeforeHandler()
    miner : Miner = miner_factory(plugins=[plugin])
    miner.call_hook_func('before_epoch_start')
    miner.call_hook_func('before_epoch_start')

    miner.call_hook_func('after_epoch_end')

    miner.call_hook_func('after_train_iteration_end')
    miner.call_hook_func('after_train_iteration_end')

    assert plugin.called_times['before_epoch_start'] == 2
    assert plugin.called_times['after_epoch_end'] == 1
    assert plugin.called_times['after_train_iteration_end'] == 0


def test_plugin_event(miner_factory):
    plugin = PluginWithTriggeredEvent()
    miner : Miner = miner_factory(plugins=[plugin])

    payloads = []
    events = []
    plugins = []

    def callback(payload, event, plugin):
        payloads.append(payload)
        events.append(event)
        plugins.append(plugin)

    plugin.on("SOME_EVENT", callback)
    miner.on("SOME_OTHER_EVENT", callback)

    miner.call_hook_func("after_epoch_end")

    assert len(payloads) == 2

    assert "key" in payloads[0]
    assert payloads[0]["key"] == "value"
    assert events[0] == "SOME_EVENT"
    assert plugins[0] == plugin

    assert "other_key" in payloads[1]
    assert payloads[1]["other_key"] == "other_value"
    assert events[1] == "SOME_OTHER_EVENT"
    assert plugins[1] == plugin
