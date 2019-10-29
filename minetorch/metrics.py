
import torch
import functools


def metric_utils(logits, targets, threshold=0.5):
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
    logits = logits > threshold
    targets = targets > 0.5
    return logits, targets


def iou(logits, targets, threshold=0.5, separate_class=False):
    logits, targets = metric_utils(logits, targets, threshold=threshold)
    intersection = (logits & targets).double().sum((1, 2))
    union = (logits | targets).double().sum((1, 2))
    iou = (intersection + 1e-7) / (union + 1e-7)
    if separate_class:
        iou = iou
    else:
        iou = iou.mean()
    return iou


single_class_iou = functools.partial(iou, separate_class=False)
multi_class_iou = functools.partial(iou, separate_class=True)


def dice(logits, targets, threshold=0.5, separate_class=False):
    logits, targets = metric_utils(logits, targets, threshold=threshold)
    intersection = (logits & targets).double().sum((1, 2))
    A = logits.double().sum((1, 2))
    B = targets.double().sum((1, 2))
    dice = (2 * intersection + 1e-7) / (A + B + 1e-7)
    if separate_class:
        dice = dice
    else:
        dice = dice.mean()
    return dice


single_class_dice = functools.partial(dice, separate_class=False)
multi_class_dice = functools.partial(dice, separate_class=True)


def accuracy(logits, targets, threshold=0.5, separate_class=False):
    logits, targets = metric_utils(logits, targets, threshold=threshold)
    true_prediction = (~(logits ^ targets)).double().sum((1, 2))
    total = logits.view(logits.shape[0], -1).shape[1]
    acc = true_prediction / total
    if separate_class:
        acc = acc
    else:
        acc = acc.mean()
    return acc


single_class_accuracy = functools.partial(accuracy, separate_class=False)
multi_class_accuracy = functools.partial(accuracy, separate_class=True)


def precision(logits, targets, threshold=0.5, separate_class=False):
    smooth = 1e-7
    logits, targets = metric_utils(logits, targets, threshold=threshold)
    tp = ((logits == 1) & (targets == 1)).double().sum((1, 2))
    fp = ((logits == 1) & (targets == 0)).double().sum((1, 2))
    prec = tp / (tp + fp + smooth)
    return prec


single_class_precission = functools.partial(precision, separate_class=False)
multi_class_precission = functools.partial(precision, separate_class=True)


def recall(logits, targets, threshold=0.5, separate_class=False):
    smooth = 1e-7
    logits, targets = metric_utils(logits, targets, threshold=threshold)
    tp = ((logits == 1) & (targets == 1)).double().sum((1, 2))
    fn = ((logits == 0) & (targets == 1)).double().sum((1, 2))
    recall = tp / (tp + fn + smooth)
    return recall


single_class_recall = functools.partial(recall, separate_class=False)
multi_class_recall = functools.partial(recall, separate_class=True)