import torch
import torch.nn.functional as F
from torch.autograd import Variable


# Below loss functions are referenced at https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
def dice_loss(inputs, targets, smooth=1):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
    return 1 - dice


def dicebce_loss(inputs, targets, smooth=1):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
    bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
    dice_bce = bce + dice_loss
    return dice_bce


def iou_loss(inputs, targets, smooth=1):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection
    iou = (intersection + smooth)/(union + smooth)
    return 1 - iou


def focal_loss(inputs, targets, alpha=0.8, gamma=2, smooth=1):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
    bce_exp = torch.exp(-bce)
    focal_loss = alpha * (1-bce_exp)**gamma * bce
    return focal_loss


def tversky_loss(inputs, targets, smooth=1, alpha=0.5, beta=0.5):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    tp = (inputs * targets).sum()
    fp = ((1-targets) * inputs).sum()
    fn = (targets * (1-inputs)).sum()
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky


def focaltversky_loss(inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    tp = (inputs * targets).sum()
    fp = ((1-targets) * inputs).sum()
    fn = (targets * (1-inputs)).sum()
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    focaltversky = (1 - tversky)**gamma
    return focaltversky


def lovaszhingeLoss(inputs, targets):
    inputs = torch.sigmoid(inputs)
    lovasz = lovasz_hinge(inputs, targets, per_image=False)
    return lovasz


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between - infty and + infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(
            lovasz_hinge_flat(
                *flatten_binary_scores(
                    log.unsqueeze(0),
                    lab.unsqueeze(0),
                    ignore)
                    ) for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(
            *flatten_binary_scores(
                logits,
                labels,
                ignore)
                )
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between - infty and + infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(
            lovasz_softmax_flat(
                *flatten_probas(
                    prob.unsqueeze(0),
                    lab.unsqueeze(0),
                    ignore
                    ),
                classes=classes) for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def combo_loss(inputs, targets, smooth=1, alpha=0.5, ce_ratio=0.5):
    e = 1e-7
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    inputs = torch.clamp(inputs, e, 1.0 - e)
    out = - (alpha * ((targets * torch.log(inputs)) + ((1 - alpha) * (1.0 - targets) * torch.log(1.0 - inputs))))
    weighted_ce = out.mean(-1)
    combo = (ce_ratio * weighted_ce) - ((1 - ce_ratio) * dice)
    return combo
