from minetorch import optimizer, option
from torch.optim import SGD

@optimizer('PyTorch SGD', 'Simple wrap of the SGD optimizer of PyTorch')
@option('lr', help='Learning Rate', type='number', required=True)
def sgd(lr):
    # TODO: how to get model parameters?
    return SGD([], lr)

