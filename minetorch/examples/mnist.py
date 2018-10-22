import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

sys.path.append('..')

from logger import Logger
from trainer import Trainer


# step 1: define some model
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


# step 2: create dataloader
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=256, shuffle=True)

val_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=256, shuffle=True)


# step 3(plan-A): define a loss computing function
def compute_loss(trainer, data):
    image, target = data
    output = trainer.model(image)
    return F.nll_loss(output, target)


# step 3(plan-B): define a loss computing function
# Uncoment the code bellow to train mnist with cross_entropy loss
# def compute_loss(model, data, logger):
#     image, target = data
#     output = model(image)
#     return F.cross_entropy(output, target)


# step 3.5(optional): define a after_epoch_end hook to compute accuracy,
#                     need some understanding of Trainer class
def after_epoch_end(trainer):
    trainer.model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in trainer.val_dataloader:
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().numpy()
    accuracy = correct / len(trainer.val_dataloader.dataset)
    trainer.logger.scalar(accuracy, trainer.current_epoch, 'accuracy')

# step 4: start to train
model = Net()

trainer = Trainer(
    logger=Logger(log_dir='./log', namespace='plan-A'),
    # if training with plan-B, use the code bellow
    # logger=Logger(log_dir='./log', namespace='plan-B'),
    model=model,
    optimizer=optim.SGD(model.parameters(), lr=0.01),
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    loss_func=compute_loss,
    hooks={'after_epoch_end': after_epoch_end}
)

trainer.train()
