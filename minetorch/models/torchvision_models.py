from minetorch import model, option
from torchvision import models
from torch import nn
import importlib
import torch.nn.functional as F

@model('Torchvision ResNet', 'Simple wrap for torchvisions ResNet')
@option('size', help='Size of ResNet', type='select', options=['18', '34', '50', '101', '152'], required=True)
@option('class_num', help='Classification class number', type='number', default=1000)
def resnet(size, class_num):
    return getattr(models, f'resnet{size}')()

@model('Torchvision VggNet', 'Simple wrap for torchvision VggNet')
@option('size', help='Size of VggNet', type='select', options=['11', '11_bn', '13', '13_bn', '16', '16_bn', '19', '19_bn'], required=True)
@option('class_num', help='Classification class number', type='number', default=100)
def vggnet(size, class_num):
    return getattr(models, f'vgg{size}')()

@model('Simple Demo Net', 'A very simple net for demo')
def demo():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    return Net()
