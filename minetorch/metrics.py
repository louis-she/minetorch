
import torch


class dice(object):
    
    def __init__(self, logits, targets, threshold=0.5, separate_class=False):
        self.logits = logits
        self.targets = targets
        self.separate_class = separate_class
        self.threshold = threshold
        assert(self.logits.shape == self.targets.shape)
        self.classes = self.logits.shape[1]

    def compute_batch(self):
        dices = []
        dice_cls = 0
        if self.separate_class == True:
             for cls in range(self.classes):
                dice_cls = dice_single_class_batch(self.logits[:,cls], self.targets[:,cls], self.threshold)
                dices.append(dice_cls)           
        else:
            for cls in range(self.logits.shape[1]):
                dice_cls += dice_single_class_batch(self.logits[:,cls], self.targets[:,cls], self.threshold)
            dices.append(dice_cls/self.classes)

        return dices


class iou(object):
    
    def __init__(self, logits, targets, threshold=0.5, separate_class=False):
        self.logits = logits
        self.targets = targets
        self.separate_class = separate_class
        self.threshold = threshold
        assert(self.logits.shape == self.targets.shape)
        self.classes = self.logits.shape[1]

    def compute_batch(self):
        ious = []
        iou_cls = 0
        if self.separate_class == True:
            for cls in range(self.classes):
                iou_cls = iou_single_class_batch(self.logits[:,cls], self.targets[:,cls], self.threshold)
                ious.append(iou_cls)
        else:
            for cls in range(self.logits.shape[1]):
                iou_cls += iou_single_class_batch(self.logits[:,cls], self.targets[:,cls], self.threshold)
            ious.append(iou_cls/self.classes)
        
        return ious


def iou_single_class_batch(logits, targets, threshold):
    
    # be with the BATCH x H x W shape
    logits = torch.sigmoid(logits.squeeze(1))  # BATCH x 1 x H x W => BATCH x H x W
    logits = logits > torch.Tensor([threshold])
    
    targets = targets.squeeze(1) > torch.Tensor([0.5])
    
    intersection = (logits & targets).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (logits | targets).float().sum((1, 2))         # Will be zero if both are 0
    
    iou = (intersection + 1e-7) / (union + 1e-7)
    
    return iou.mean()

def dice_single_class_batch(logits, targets, threshold):
    
    # be with the BATCH x H x W shape
    logits = torch.sigmoid(logits.squeeze(1))  # BATCH x 1 x H x W => BATCH x H x W
    logits = logits > torch.Tensor([threshold])
    
    targets = targets.squeeze(1) > torch.Tensor([0.5])
    
    intersection = (logits & targets).double().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    A = logits.double().sum()         # Will be zero if both are 0
    B = targets.double().sum()
    dice = 2*(intersection + 1e-7) / (A + B + 1e-7)
    
    return dice.mean()