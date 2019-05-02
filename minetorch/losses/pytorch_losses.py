from minetorch import loss, option
import torch.nn.functional as F

@loss('PyTorch cross_entropy')
def cross_entropy():
    def loss(input, target):
        F.cross_entropy(input, target)
    return loss
