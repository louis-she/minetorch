
import torch
import functools


def shape_norm(logits, targets):
    # Expect one of the following shapes [classes, height, width], [height, width], [classes].
    assert logits.shape == targets.shape, "Inputs not match with targets on shape"
    assert len(logits.shape) <= 3, 'Expect one of the following shapes [classes, height, width], [height, width], [classes].'
    if len(logits.shape) == 1:
        logits = logits.unsqueeze(1).unsqueeze(1)
        targets = targets.unsqueeze(1).unsqueeze(1)
    elif len(logits.shape) == 2:
        logits = logits.unsqueeze(0)
        targets = targets.unsqueeze(0)
    logits = torch.sigmoid(logits)
    return logits, targets


def compute_mse(logits, targets, separate_class=False):
    logits, targets = shape_norm(logits, targets)
    error = (abs(logits - targets) ** 2).double().sum((1, 2))
    total = logits.view(logits.shape[0], -1).shape[1]
    mse = error / total
    if separate_class:
        mse = mse
    else:
        mse = mse.mean()
    return mse


def compute_mae(logits, targets, separate_class=False):
    logits, targets = shape_norm(logits, targets)
    error = (abs(logits - targets)).double().sum((1, 2))
    total = logits.view(logits.shape[0], -1).shape[1]
    mae = error / total
    if separate_class:
        mae = mae
    else:
        mae = mae.mean()
    return mae


def confusion_matrix(logits, targets, threshold=0.5, separate_class=False, func=lambda x: x):
    logits, targets = shape_norm(logits, targets)
    logits = logits > threshold
    targets = targets > 0.5
    tp = ((logits == 1) & (targets == 1)).double().sum((1, 2)).unsqueeze(0)
    fp = ((logits == 1) & (targets == 0)).double().sum((1, 2)).unsqueeze(0)
    fn = ((logits == 0) & (targets == 1)).double().sum((1, 2)).unsqueeze(0)
    tn = ((logits == 0) & (targets == 0)).double().sum((1, 2)).unsqueeze(0)
    return torch.cat((tp, fp, fn, tn), 0), func


def compute_precision(c_matrix):
    return c_matrix[0] / (c_matrix[0] + c_matrix[1])


def compute_recall(c_matrix):
    return c_matrix[0] / (c_matrix[0] + c_matrix[2])


def compute_accuracy(c_matrix):
    return (c_matrix[0] + c_matrix[3]) / (c_matrix[0] + c_matrix[1] + c_matrix[2] + c_matrix[3])


def compute_dice(c_matrix):
    smooth = 1e-7
    return (2 * c_matrix[0] + smooth) / (2 * c_matrix[0] + c_matrix[1] + c_matrix[2] + smooth)


def compute_iou(c_matrix):
    smooth = 1e-7
    return (c_matrix[0] + smooth) / (c_matrix[0] + c_matrix[1] + c_matrix[2] + smooth)


precision = functools.partial(confusion_matrix, func=compute_precision, separate_class=True)
recall = functools.partial(confusion_matrix, func=compute_recall, separate_class=True)
accuracy = functools.partial(confusion_matrix, func=compute_accuracy, separate_class=True)
dice = functools.partial(confusion_matrix, func=compute_dice, separate_class=True)
iou = functools.partial(confusion_matrix, func=compute_iou, separate_class=True)
