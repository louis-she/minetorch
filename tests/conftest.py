import pytest
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from minetorch.charts import Chart, NoneChart
from minetorch.miner import Miner
from minetorch.plugin import Plugin


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


class FixedDataset(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class MockChart(Chart):

    def init(self):
        self.points = {}

    def add_points(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.points:
                self.points[k] = []
            self.points[k].append(v)


@pytest.fixture
def data_loader():

    class FakeDataset(Dataset):

        def __getitem__(self, _):
            return torch.rand(1, 28, 28), (torch.rand(1) * 10).long().squeeze()

        def __len__(self):
            return 12

    return DataLoader(FakeDataset(), batch_size=2)


@pytest.fixture()
def miner_factory(tmpdir_factory, data_loader):
    base_dir = tmpdir_factory.mktemp('base_dir')

    def forward(miner, data):
        loss = miner.model(torch.rand(2, 10)).mean()
        return data[0], loss

    def func(**kwargs):
        model = Net()
        params = {
            "base_dir": base_dir,
            "code": "test",
            "model": model,
            "optimizer": optim.SGD(model.parameters(), lr=0.01),
            "train_dataloader": data_loader,
            "val_dataloader": data_loader,
            "forward": forward,
            "chart_type": MockChart,
            "max_epochs": 1,
            **kwargs
        }
        return Miner(**params)

    return func


@pytest.fixture()
def dataloader_factory():
    def create_fixed_dataloader(data, labels, **kwargs):
        return DataLoader(FixedDataset(data, labels), **kwargs)
    return create_fixed_dataloader
