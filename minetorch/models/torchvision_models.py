from minetorch import model, option
from torchvision import models
import importlib

@model('Torchvision ResNet', 'Simple wrap for torchvisions ResNet')
@option('size', help='Size of ResNet', type='select', options=['18', '34', '50', '101', '152'], required=True)
@option('class_num', help='Classification class number', type='number', default=1000)
def resnet(size, class_num):
    return getattr(models, f'resnet{size}')()

@model('Torchvision VggNet', 'Simple wrap for torchvision VggNet')
@option('size', help='Size of VggNet', type='select', options=['11', '11_bn', '13', '13_bn', '16', '16_bn', '19', '19_bn'], required=True)
@option('class_num', help='Classification class number', type='number', default=100)
def vggnet(size):
    return getattr(models, f'vgg{size}')()
