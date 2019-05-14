from minetorch import optimizer, option, g
from torch.optim import SGD

@optimizer('PyTorch SGD', 'Simple wrap of the SGD optimizer of PyTorch')
@option('lr', help='Learning Rate', type='number', required=True, default='0.1')
def sgd(lr):
    # TODO: how to get model parameters?
    return SGD(g.model.parameters(), float(lr))
