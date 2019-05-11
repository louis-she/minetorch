from minetorch import loss, option
import torch.nn.functional as F

@loss('PyTorch cross_entropy', 'Simple wrap of the cross_entropy of PyTorch')
def cross_entropy():
    def loss(input, target):
        F.cross_entropy(input, target)
    return loss
