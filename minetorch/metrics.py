import operator
from typing import Union, Callable
import numpy as np
import torch

from minetorch.plugin import RecordOutput, Plugin


class Metric(Plugin):
    """Metric base class, all the metric should be inherited from.

    Args:
        optim can be both str or a Callable, if it's a str, it can be ">" or "max", "<" or "min";
        if it's a Callable, it should have a 2 argument, the first is the latest score and the second
        is the best score, return True if the latest score beats the best or other wise.
    """

    __state_members__ = [
        "last_score",
        "best_score",
        "optim_func",
        "best_score_epoch",
        "scores",
    ]

    __events__ = ["NEW_BEST_SCORE"]

    def __init__(self, optim=Union[Callable, str], **kwargs):
        super().__init__(**kwargs)

        if optim == ">" or optim == "max":
            self.optim_func = operator.gt
            self.best_score = -float("inf")
        elif optim == "<" or optim == "min":
            self.optim_func = operator.lt
            self.best_score = float("inf")
        else:
            self.optim_func = optim

        self.last_score = None
        self.scores = []

    def add_score(self, score: float):
        self.scores.append(score)
        self.last_score = score

        if self.optim_func(self.last_score, self.best_score):
            self.notify(
                f"New best score {self.last_score} beats the last best {self.best_score}",
                "success",
            )
            self.best_score = self.last_score
            self.trigger("NEW_BEST_SCORE")


class Accuracy(RecordOutput, Metric):
    """Accuracy metric calculate the accuracy of every sample, then
    average overall samples. So the metric accept 2 arrays of the same
    shape of [samples_num, *].

    But in most of the time, the modal's outputs and labels are not
    the same as this metric expected. The data should always be transformed
    before feed into the metric calculator. So for convinience, the metric
    has some built-in transforms based on the most use case:

    built-in transform:
        - `cross_entropy_transform` the input should be the same as PyTorch's `F.cross_entropy`
        - `binary_cross_entropy_with_logits_transform` the input should be the same as PyTorch's
          `F.binary_cross_entropy_with_logits`

    feel free to add a fully customied transform function, the transform function
    should take 2 arguments:
        the first is the raw output data from the modal(logits)
        the second is the labels yields from DataLoader

    There is no option for customized input_type, if one has more complicated needs,
    just inherite this class and overwrite the `after_epoch_end` medhod.
    """

    def __init__(
        self,
        transform: Union[str, Callable] = "cross_entropy_transform",
        target: Union[Callable, int, str] = 1,
    ):
        super().__init__(target=target, optim="max")
        self.transform_func = getattr(self, transform)

    def after_init(self):
        self.chart = self.miner.create_chart("accuracy")

    @staticmethod
    def cross_entropy_transform(raw_outputs, labels):
        return torch.argmax(raw_outputs, axis=1), labels

    @staticmethod
    def binary_cross_entropy_with_logits_transform(raw_outputs, labels):
        return raw_outputs > 0, labels

    def accuracy_of_tensors(self, a, b):
        """get accuracy of two tensors, the shape of both should be
        [N, *], where N is the number of samples and * can be any shape.
        """
        equals = a == b
        if len(equals.shape) == 1:
            return equals.sum() / len(equals)
        else:
            acc_of_samples = (
                equals.sum(dim=list(range(1, len(a.shape)))) / a.shape[1:].numel()
            )
            return acc_of_samples.mean()

    def after_epoch_end(self):
        super().after_epoch_end()
        val_score = self.accuracy_of_tensors(
            *self.transform_func(self.val_raw_outputs, self.val_labels)
        )
        train_score = self.accuracy_of_tensors(
            *self.transform_func(self.train_raw_outputs, self.train_labels)
        )
        self.add_score(val_score)
        self.chart.add_points(train_acc=train_score, val_acc=val_score)
