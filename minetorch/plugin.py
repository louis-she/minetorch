import os
from typing import Any, Dict
import typing
import numpy as np
import functools
import operator

import torch

_getitem = lambda a, b: operator.getitem(a, b)

class Plugin:
    """Plugin is a self-contained, statable object for adding
    extra function to MineTorch, it's a more solid way than hook
    functions.
    """
    # The state to persist for each `miner.persist` call,
    __state_members__ = []

    def __init__(self, prefix: str=""):
        self.name = self.__class__.__name__
        self.miner = None
        self.prefix = prefix
        self.event_handlers = {}

    def load_state_dict(self, state: Dict):
        for key in state:
            setattr(self, key, state[key])

    def state_dict(self):
        state = {}
        for key in self.__state_members__:
            value = getattr(self, key, None)
            state[key] = value
        return state

    def set_miner(self, miner):
        self.miner = miner

    def notify(self, message: str, _type : str):
        message = f"[{self.name}] {message}"
        self.miner.notify(message, _type)

    def __getattr__(self, key):
        if self.miner is None or not hasattr(self.miner, key):
            raise AttributeError(key)
        return getattr(self.miner, key)

    def print_txt(self, printable : Any, name: str):
        """write information to files every epoch
        """
        with open(self.plugin_file(f"{name}.txt"), "a") as f:
            print(
                f"================ Epoch {self.current_epoch} ================\n",
                file=f,
            )
            print(printable, file=f)
            print("\n\n", file=f)

    @property
    def plugin_dir(self):
        if hasattr(self, "_plugin_dir"):
            return getattr(self, "_plugin_dir")

        plugin_dir = os.path.join(self.code_dir, self.__class__.__name__)
        try:
            os.mkdir(plugin_dir)
        except FileExistsError:
            pass
        self._plugin_dir = plugin_dir
        return self._plugin_dir

    def plugin_file(self, name):
        return os.path.join(self.plugin_dir, name)

    def on(self, event: str, callback: typing.Callable):
        """Listen on the event of this plugin.
        Plugin should statement the event it will trigger
        with __event__ variable
        """
        if event not in self.__events__:
            raise RuntimeError(f"Event {event} is not valid in {self.__class__.__name__}")
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(callback)

    def trigger(self, event: str, payload: typing.Any):
        """Plugin trigger some event with payload
        """
        for callback in self.event_handlers[event]:
            callback(payload)

    def before_handler(self, hook_point: str, payload: Any):
        """If defined, every hook function will be called after
        this `before_handler`, if this function return True, then
        the follling hook function will be called, if False,
        then the following function will be ignored, much like a
        switch of the hook function.
        """
        return True

    def before_init(self):
        """Before init the plugin and models, the `self.miner` object
        can be used at this time, but is not init yet.

        Typical usage: want to change the static parameters passing to the
        miner, so we can modified the init action. It's rare so this hook
        is not used so much.
        """

    def after_init(self):
        """After init the plugin and models, the `self.miner` and all the
        plugins are inited. Be awared that the persisted state is applyed
        so some of the member vairable should have values now, if assign
        new value, the state value will be overrided.

        Typical usage: init things that are associated with the states.
        """

    def after_init(self):
        """After init the plugin and models, the `self.miner` and all the
        plugins are inited. Be awared that the persisted state is applyed
        so some of the member vairable should have values now, if assign
        new value, the state value will be overrided.

        Typical usage: init things that are associated with the states.
        """

    def before_epoch_start(self):
        """Before every epoch started. Do things that are associated with
        every epoch.

        Typical usage: some metric plugin would like init the metric value
        to zeros here.
        """

    def before_train_iteration_start(self, data: Any, index: int):
        """Before every train iteration started.

        Args:
            data: the batch samples yield by dataloader
            index: index of this batch in the dataloader

        The arguments is just like:

        >>> for (index, data) in enumerate(self.miner.train_dataloader): ...
        """

    def after_train_iteration_end(self, data: Any, index: int, loss: torch.Tensor, predicts: torch.Tensor):
        """After every train iteration ended.

        Args:
            data: the batch samples yield by dataloader
            index: index of this batch in the dataloader

        The arguments is just like:

        >>> for (index, data) in enumerate(self.miner.train_dataloader): ...
        """

    def before_val_iteration_start(self, data: Any, index: int):
        """Before every validation iteration started.

        Args:
            data: the batch samples yield by dataloader
            index: index of this batch in the dataloader

        The arguments is just like:

        >>> for (index, data) in enumerate(self.miner.val_dataloader): ...
        """

    def after_val_iteration_start(self, data: Any, index: int):
        """Before every validation iteration ended.

        Args:
            data: the batch samples yield by dataloader
            index: index of this batch in the dataloader

        The arguments is just like:

        >>> for (index, data) in enumerate(self.miner.val_dataloader): ...
        """

    def after_epoch_end(self):
        """After current epoch finished.

        Typicall usage: for metric plugin, this should be a good place
        to report the final score.
        """

    def before_quit(self):
        """Before all the training epoch finished, right before the
        process existed.
        """


class RecordOutput:
    """A mixin that will collect the model's outputs and the coorsponding
    labels in every epoch. The Plugin with this mixin can access to the
    following members:

    `self.train_raw_outputs` np.Array, shape is [train_samples, *]
        should be the raw outputs(forward first returned value)
        from the modal
    `self.train_labels` np.Array, shape is [train_samples, *]
        should be the labels that yield from dataset
    `self.val_raw_outputs` same as `self.train_raw_outputs` but data
        is from val
    `self.val_labels` same as `self.train_labels` but data is from val

    Args:
        target how to get target from data that yield from DataLoader
    """
    def __init__(self, target: typing.Union[typing.Callable, int, str] = None, **kwargs):
        self._get_target_func = target
        if self._get_target_func is None:
            self._get_target_func = functools.partial(_getitem, b=1)
        if isinstance(self._get_target_func, [int, str]):
            self._get_target_func = functools.partial(_getitem, b=self.target)
        super().__init__(**kwargs)

    def before_epoch_start(self):
        self.train_raw_outputs = []
        self.train_labels = []
        self.val_raw_outputs = []
        self.val_labels = []

    def _iteration_end(self, phase, raw_outputs, data):
        if phase == "train":
            target_raw_outputs = self.train_raw_outputs
            target_labels = self.train_labels
        else:
            target_raw_outputs = self.val_raw_outputs
            target_labels = self.val_labels

        raw_outputs = raw_outputs.detach().cpu().numpy()
        targets = self._get_target_func(data).cpu().numpy()
        target_raw_outputs.append(raw_outputs)
        target_labels.val_labels.append(targets)

    def after_train_iteration_end(self, raw_outputs, data, **ignore):
        self._iteration_end("train", raw_outputs, data)

    def after_val_iteration_end(self, raw_outputs, data, **ignore):
        self._iteration_end("val", raw_outputs, data)

    def after_epoch_end(self):
        self.train_raw_outputs = np.concatenate(self.train_raw_outputs)
        self.train_labels = np.concatenate(self.train_labels)
        self.val_raw_outputs = np.concatenate(self.val_raw_outputs)
        self.val_labels = np.concatenate(self.val_labels)