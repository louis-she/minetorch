import pytest
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from minetorch.charts import NoneChart
from minetorch.miner import Miner
from minetorch.plugin import Plugin


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


@pytest.fixture
def data_loader():

    class FakeDataset(Dataset):

        def __getitem__(self, _):
            return torch.rand(1, 28, 28), (torch.rand(1) * 10).long().squeeze()

        def __len__(self):
            return 10

    return DataLoader(FakeDataset(), batch_size=2)


@pytest.fixture()
def miner_factory(tmpdir_factory, data_loader):
    base_dir = tmpdir_factory.mktemp('base_dir')

    def func(**kwargs):
        model = Net()
        return Miner(
            base_dir=base_dir,
            code="test",
            model=model,
            optimizer=optim.SGD(model.parameters(), lr=0.01),
            train_dataloader=data_loader,
            val_dataloader=data_loader,
            loss_func=torch.nn.CrossEntropyLoss(),
            chart_type=NoneChart,
            **kwargs
        )

    return func
