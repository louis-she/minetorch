import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

from .plugin import Plugin


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

    def before_init(self):
        self.create_sheet_column("latest_confusion_matrix", "Latest Confusion Matrix")
        self.create_sheet_column("kappa_score", "Kappa Score")
        self.create_sheet_column("accuracy", "Accuracy")

    def before_epoch_start(self, epoch):
        self.raw_output = []
        self.predicts = np.array([]).astype(np.float)
        self.targets = np.array([]).astype(np.int)

    def after_val_iteration_ended(self, predicts, data, **ignore):
        raw_output = predicts.detach().cpu().numpy()
        predicts = np.argmax(raw_output, axis=1)
        predicts = predicts.reshape([-1])
        targets = data[1].cpu().numpy().reshape([-1])

        self.raw_output.append(raw_output)
        self.predicts = np.concatenate((self.predicts, predicts))
        self.targets = np.concatenate((self.targets, targets))

    def after_epoch_end(self, val_loss, **ignore):
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
        png_file = self.scalars(
            {"accuracy": (self.predicts == self.targets).sum() / len(self.predicts)},
            "accuracy",
        )
        if png_file:
            self.update_sheet(
                "accuracy", {"raw": png_file, "processor": "upload_image"}
            )

    def _confusion_matrix(self):
        matrix = confusion_matrix(self.targets, self.predicts)
        self.print_txt(matrix, "confusion_matrix")

    def _kappa_score(self):
        png_file = self.scalars(
            {
                "kappa_score": cohen_kappa_score(
                    self.targets, self.predicts, weights="quadratic"
                )
            },
            "kappa_score",
        )

        if png_file:
            self.update_sheet(
                "kappa_score", {"raw": png_file, "processor": "upload_image"}
            )

    def _save_results(self):
        file_name = self.plugin_file(f"result.{self.current_epoch}.npz")
        raw_output = np.stack(self.raw_output)
        np.savez_compressed(file_name, predicts=self.predicts, targets=self.targets, raw_output=raw_output)
