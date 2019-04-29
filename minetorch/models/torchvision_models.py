
__package__ = 'models'
from minetorch import model, option
from torchvision import models
import importlib

@model('Torchvision ResNet')
@option('size', help='Size of ResNet', type='enum', choices=[18, 34, 50, 101, 152], required=True)
@option('class_num', help='Classification class number', type='number')
def resnet(size):
    return getattr(models, f'resnet{size}')()

@model('Torchvision VggNet')
@option('size', help='Size of VggNet', type='enum', choices=['11', '11_bn', '13', '13_bn', '16', '16_bn', '19', '19_bn'], required=True)
@option('class_num', help='Classification class number', type='number', default=100)
def vggnet(size):
    return getattr(models, f'vgg{size}')()
