import torch
from typing import Any


class {{cookiecutter.class_name}}:

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

    def after_train_iteration_end(
        self, data: Any, index: int, loss: torch.Tensor, raw_outputs: torch.Tensor
    ):
        """After every train iteration ended.

        Args:
            data: the batch samples yield by dataloader
            index: index of this batch in the dataloader
            loss: loss of this batch
            raw_outputs: outputs from model
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
        """Before every validation iteration started.

        Args:
            data: the batch samples yield by dataloader
            index: index of this batch in the dataloader

        The arguments is just like:

        >>> for (index, data) in enumerate(self.miner.val_dataloader): ...
        """

    def after_val_iteration_end(
        self, data: Any, index: int, loss: torch.Tensor, raw_outputs: torch.Tensor
    ):
        """Before every validation iteration ended.

        Args:
            data: the batch samples yield by dataloader
            index: index of this batch in the dataloader
            loss: loss of this batch
            raw_outputs: outputs from model
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