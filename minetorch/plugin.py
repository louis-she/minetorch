import os
from typing import Any, Dict

import torch


class Plugin:
    """Plugin is a self-contained, statable object for adding
    extra function to MineTorch, it's a more solid way than hook
    functions.
    """

    # The state to persist for each `miner.persist` call
    # by default all the variables are counted as states
    __state_members__ = None

    def __init__(self, prefix: str=""):
        self.name = self.__class__.__name__
        self.miner = None
        self.prefix = prefix

    def load_state_dict(self, state: Dict):
        for key in state:
            if hasattr(self, key):
                self.notify(f"Overrting {key} of plugin {self.__class__.__name__} by passed in state", "warning")
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

    def create_sheet_column(self, key, name):
        self.miner.create_sheet_column(f"{self.prefix}{key}", f"{self.prefix}{name}")

    def update_sheet(self, key, value):
        self.miner.update_sheet(f"{self.prefix}{key}", value)

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
