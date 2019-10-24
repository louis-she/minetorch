
import torch

def iou(logits, targets, threshold, separate_class=False): 
    # be with the C x H x W shape
    logits = torch.sigmoid(logits)
    logits = logits > torch.Tensor([threshold])
    targets = targets > torch.Tensor([0.5])
    
    intersection = (logits & targets).float().sum((1, 2))
    union = (logits | targets).long().sum((1, 2))
    
    iou = (intersection + 1e-7) / (union + 1e-7)
    if separate_class:
        iou = iou
    else:
        iou = iou.mean()
    return iou

def dice(logits, targets, threshold, separate_class=False):
    
    # be with the C x H x W shape
    logits = torch.sigmoid(logits)  # BATCH x 1 x H x W => BATCH x H x W
    logits = logits > torch.Tensor([threshold])
    targets = targets > torch.Tensor([0.5])
    
    intersection = (logits & targets).long().sum((1, 2))
    A = logits.long().sum()
    B = targets.long().sum()
    dice = 2*(intersection + 1e-7) / (A + B + 1e-7)
    if separate_class:
        dice = dice
    else:
        dice = dice.mean()
    return dice