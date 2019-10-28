
import torch
import functools


def iou(logits, targets, threshold=0.5, separate_class=False):
    # be with the C x H x W shape
    logits = torch.sigmoid(logits)
    logits = logits > threshold
    targets = targets > 0.5
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
    # be with the C x H x W shape
    logits = torch.sigmoid(logits)
    logits = logits > threshold
    targets = targets > 0.5
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
    # be with the C x H x W shape
    logits = torch.sigmoid(logits)
    logits = logits > threshold
    targets = targets > 0.5

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
