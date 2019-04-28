
__package__ = 'models'
from minetorch.core import model, option, Choice
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

@model
@option('size', help='Size of ResNet', type=Choice([18, 34, 50, 101, 152]))
def resnet(name, parameters):
    pass
