from ast import operator
from typing import Union, Callable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

from minetorch.plugin import RecordOutput, Plugin
from sklearn.metrics import accuracy_score


class Metric(Plugin):

    __state_members__ = [
        "last_score",
        "best_score",
        "optim_func",
        "best_score_epoch",
        "scores",
    ]

    __events__ = ["NEW_BEST_SCORE"]

    def __init__(self, optim=Union[Callable, str], **kwargs):
        """Metric base class, all the metric should be inherited from.

        Args:
            optim can be both str or a Callable, if it's a str, it can be ">" or "max", "<" or "min";
            if it's a Callable, it should have a 2 argument, the first is the latest score and the second
            is the best score, return True if the latest score beats the best or other wise.
        """
        super().__init__(**kwargs)

        if optim == ">" or optim == "max":
            self.optim_func = operator.gt
        elif optim == "<" or optim == "min":
            self.optim_func = operator.lt
        else:
            self.optim_func = optim

        self.best_score = float("inf")
        self.last_score = None
        self.scores = []

    def add_score(self, score: float):
        self.scores.append(score)
        self.last_score = score

        if self.optim_func(self.last_score, self.best_score):
            self.notify(
                f"New best score {self.last_score} beats the last best {self.best_score}"
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
        - `ce` the input should be the same as PyTorch's `F.cross_entropy`
        - `bce` the input should be the same as PyTorch's `F.binary_cross_entropy_with_logits`

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
        self.transform_func = getattr(self, transform, transform)

    @staticmethod
    def cross_entropy_transform(raw_outputs, labels):
        return np.argmax(raw_outputs, axis=1), labels

    @staticmethod
    def binary_cross_entropy_with_logits_transform(raw_outputs, labels):
        return raw_outputs > 0, labels

    def after_epoch_end(self):
        super().after_epoch_end()
        self.val_raw_outputs, self.val_labels = self.transform_func(
            self.val_raw_outputs, self.val_labels
        )

        self.train_raw_outputs, self.train_labels = self.transform_func(
            self.train_raw_outputs, self.train_labels
        )


class MultiClassesClassificationMetricWithLogic(Plugin):
    """MultiClassesClassificationMetric
    This can be used directly if your loss function is torch.nn.CrossEntropy
    """

    def __init__(
        self,
        accuracy=True,
        confusion_matrix=True,
        kappa_score=True,
        plot_confusion_matrix=True,
        classification_report=True,
        sheet_key_prefix="",
    ):

        super().__init__(sheet_key_prefix)
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix
        self.kappa_score = kappa_score
        self.plot_confusion_matrix = plot_confusion_matrix
        self.classification_report = classification_report
        self.sheet_key_prefix = sheet_key_prefix
        self.accuracy_chart = self.create_chart(name="accuracy")
        self.kappa_score_chart = self.create_chart(name="kappa_score")

    def before_init(self):
        self.create_sheet_column("latest_confusion_matrix", "Latest Confusion Matrix")
        self.create_sheet_column("kappa_score", "Kappa Score")
        self.create_sheet_column("accuracy", "Accuracy")

    def before_epoch_start(self, epoch):
        self.raw_output = []
        self.predicts = []
        self.targets = []

    def after_val_iteration_ended(self, predicts, data, **ignore):
        raw_output = predicts.detach().cpu().numpy()
        predicts = np.argmax(raw_output, axis=1)
        targets = data[1].cpu().numpy()

        self.raw_output.append(raw_output)
        self.predicts.append(predicts)
        self.targets.append(targets)

    def after_epoch_end(self, val_loss, **ignore):
        self.predicts = np.concatenate(self.predicts)
        self.targets = np.concatenate(self.targets)

        self._save_results()
        self.accuracy and self._accuracy()
        self.confusion_matrix and self._confusion_matrix()
        self.kappa_score and self._kappa_score()
        self.classification_report and self._classification_report()
        self.plot_confusion_matrix and self._plot_confusion_matrix(val_loss)

    def _classification_report(self):
        result = classification_report(self.targets, self.predicts)
        self.print_txt(result, "classification_report")

    def _plot_confusion_matrix(self, val_loss):
        matrix = confusion_matrix(self.targets, self.predicts)
        df_cm = pd.DataFrame(matrix)
        svm = sn.heatmap(df_cm, annot=True, cmap="OrRd", fmt=".3g")
        figure = svm.get_figure()
        if val_loss < self.lowest_val_loss:
            figure.savefig(
                self.plugin_file("confusion_matrix_epoch_best.png"), facecolor="#F0FFFC"
            )
        figure.savefig(
            self.plugin_file(f"confusion_matrix_epoch_{self.current_epoch}.png"),
            facecolor="#F0FFFC",
        )
        figure.savefig(
            self.plugin_file("confusion_matrix_epoch_latest.png"), facecolor="#F0FFFC"
        )
        plt.clf()

        self.update_sheet(
            "latest_confusion_matrix",
            {
                "raw": self.plugin_file("confusion_matrix_epoch_latest.png"),
                "processor": "upload_image",
            },
        )

    def _accuracy(self):
        score = (self.predicts == self.targets).sum() / len(self.predicts)
        self.accuracy_chart.add_points(accuracy=score)

    def _confusion_matrix(self):
        matrix = confusion_matrix(self.targets, self.predicts)
        self.print_txt(matrix, "confusion_matrix")

    def _kappa_score(self):
        score = cohen_kappa_score(self.targets, self.predicts, weights="quadratic")
        self.kappa_score_chart.add_points(kappa_score=score)

    def _save_results(self):
        file_name = self.plugin_file(f"result.{self.current_epoch}.npz")
        raw_output = np.concatenate(self.raw_output)
        np.savez_compressed(
            file_name,
            predicts=self.predicts,
            targets=self.targets,
            raw_output=raw_output,
        )
