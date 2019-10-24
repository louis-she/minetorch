
import torch

def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator

def iou(separate_class=False):
    @rename('iou')
    def compute_iou(logits, targets, threshold=0.5, separate_class_=separate_class):
        # be with the C x H x W shape
        logits = torch.sigmoid(logits)
        logits = logits > torch.Tensor([threshold])
        targets = targets > torch.Tensor([0.5])
        
        intersection = (logits & targets).double().sum((1, 2))
        union = (logits | targets).double().sum((1, 2))
        
        iou = (intersection + 1e-7) / (union + 1e-7)
        if separate_class:
            iou = iou
        else:
            iou = iou.mean()
        return iou
    return compute_iou

def dice(separate_class=False):
    @rename('dice')
    def compute_dice(logits, targets, threshold=0.5, separate_class_=separate_class):
        # be with the C x H x W shape
        logits = torch.sigmoid(logits)  # BATCH x 1 x H x W => BATCH x H x W
        logits = logits > torch.Tensor([threshold])
        targets = targets > torch.Tensor([0.5])
        
        intersection = (logits & targets).double().sum((1, 2))
        A = logits.double().sum()
        B = targets.double().sum()
        dice = 2*(intersection + 1e-7) / (A + B + 1e-7)
        if separate_class:
            dice = dice
        else:
            dice = dice.mean()
        return dice
    return compute_dice