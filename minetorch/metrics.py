
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


def compute_iou(logits, targets, threshold=0.5, separate_class=False):
    logits, targets = shape_norm(logits, targets)
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


def compute_dice(logits, targets, threshold=0.5, separate_class=False):
    logits, targets = shape_norm(logits, targets)
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


def compute_accuracy(logits, targets, threshold=0.5, separate_class=False):
    logits, targets = shape_norm(logits, targets)
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


def compute_precision(logits, targets, threshold=0.5, separate_class=False):
    smooth = 1e-7
    logits, targets = shape_norm(logits, targets)
    logits = logits > threshold
    targets = targets > 0.5
    tp = ((logits == 1) & (targets == 1)).double().sum((1, 2))
    fp = ((logits == 1) & (targets == 0)).double().sum((1, 2))
    prec = tp / (tp + fp + smooth)
    if separate_class:
        prec = prec
    else:
        prec = prec.mean()
    return prec


def compute_recall(logits, targets, threshold=0.5, separate_class=False):
    smooth = 1e-7
    logits, targets = shape_norm(logits, targets)
    logits = logits > threshold
    targets = targets > 0.5
    tp = ((logits == 1) & (targets == 1)).double().sum((1, 2))
    fn = ((logits == 0) & (targets == 1)).double().sum((1, 2))
    recall = tp / (tp + fn + smooth)
    if separate_class:
        recall = recall
    else:
        recall = recall.mean()
    return recall

iou = functools.partial(compute_iou, separate_class=False)
dice = functools.partial(compute_dice, separate_class=False)
mse = functools.partial(compute_mse, separate_class=False)
mae = functools.partial(compute_mae, separate_class=False)
accuracy = functools.partial(compute_accuracy, separate_class=False)
precission = functools.partial(compute_precision, separate_class=False)
recall = functools.partial(compute_recall, separate_class=False)


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


precision = functools.partial(confusion_matrix, func=compute_precision, separate_class=False)