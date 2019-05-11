from minetorch import loss, option
import torch.nn.functional as F

@loss('PyTorch cross_entropy', 'Simple wrap of the cross_entropy of PyTorch')
def cross_entropy():
    def loss(input, target):
        F.cross_entropy(input, target)
    return loss

@loss('PyTorch binary_cross_entropy', 'Simple wrap of the binary_cross_entropy of PyTorch')
def binary_cross_entropy():
    def loss(input, target):
        F.binary_cross_entropy(input, target)
    return loss
