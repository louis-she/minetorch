import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from .plugin import Plugin


class MultiClassesClassificationMetricWithLogic(Plugin):
    """MultiClassesClassificationMetric
    This can be used directly if your loss function is torch.nn.CrossEntropy
    """

    def __init__(self, accuracy=True, confusion_matrix=True, kappa_score=True):
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix
        self.kappa_score = kappa_score

    def before_epoch_start(self, epoch):
        self.predicts = np.array([]).astype(np.float)
        self.targets = np.array([]).astype(np.int)

    def after_val_iteration_ended(self, predicts, data, **ignore):
        predicts = np.argmax(predicts.detach().cpu().numpy(), axis=1)

        predicts = predicts.reshape([-1])
        targets = data[1].cpu().numpy().reshape([-1])

        self.predicts = np.concatenate((self.predicts, predicts))
        self.targets = np.concatenate((self.targets, targets))

    def after_epoch_end(self, **ignore):
        self.accuracy and self._accuracy()
        self.confusion_matrix and self._confusion_matrix()
        self.kappa_score and self._kappa_score()

    def _accuracy(self):
        self.drawer.scalars(
            {'accuracy': (self.predicts == self.targets).sum() / len(self.predicts)}, 'accuracy'
        )

    def _confusion_matrix(self):
        matrix = confusion_matrix(self.targets, self.predicts)
        self.print_txt(matrix, 'confusion_matrix')

    def _kappa_score(self):
        self.drawer.scalars(
            {'kappa_score': cohen_kappa_score(self.targets, self.predicts, weights='quadratic')}, 'kappa_score'
        )
