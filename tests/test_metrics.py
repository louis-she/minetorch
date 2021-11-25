import torch
from minetorch import Miner
from minetorch.metrics import Metric, Accuracy


class MockMetric(Metric):
    pass


def test_base_metric(miner_factory):
    max_metric = MockMetric(optim='max')
    min_metric = MockMetric(optim='min')
    miner_factory(plugins=[max_metric, min_metric])

    min_metric.add_score(10)
    min_metric.add_score(99)
    min_metric.add_score(95)

    max_metric.add_score(10)
    max_metric.add_score(99)
    max_metric.add_score(95)
    assert max_metric.last_score == 95
    assert max_metric.best_score == 99

    assert min_metric.last_score == 95
    assert min_metric.best_score == 10


def test_base_metric_state(miner_factory):
    metric = MockMetric(optim='max')
    miner: Miner = miner_factory(plugins=[metric])

    metric.add_score(10)
    metric.add_score(99)
    metric.add_score(95)
    miner.persist('test_metric')

    metric_resume = MockMetric(optim='max')
    miner_factory(
        plugins=[metric_resume],
        resume="test_metric"
    )

    assert metric_resume.last_score == 95
    assert metric_resume.best_score == 99


def test_ce_transform_accuracy(miner_factory, dataloader_factory):
    metric = Accuracy(transform="cross_entropy_transform")

    # simulate cross entropy logits and labels
    train_data = torch.tensor([
        [1, 8, 2],   # pred: 1
        [3, 1, 9],   # pred: 2
        [-1, 9, 2],  # pred: 1
        [10, 0, -8], # pred: 0
    ])
    train_label = torch.tensor([1, 2, 1, 1]) # acc should be: 0.75
    train_dataloader = dataloader_factory(train_data, train_label, batch_size=3)

    val_data = torch.tensor([
        [0, 2, 3],    # pred: 2
        [-2, 1, 9],   # pred: 2
        [4, 9, 2],    # pred: 1
        [9, 0, -8],   # pred: 0
        [10, 0, 12],  # pred: 2
    ])
    val_label = torch.tensor([0, 0, 1, 2, 2])  # ac should be : 0.4
    val_dataloader = dataloader_factory(val_data, val_label, batch_size=3)

    miner = miner_factory(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        plugins=[metric],
    )
    miner.train()

    assert metric.last_score == 0.4
    assert metric.best_score == 0.4
    assert metric.chart.points["train_acc"][0] == 0.75
    assert metric.chart.points["val_acc"][0] == 0.4


def test_bce_transform_accuracy(miner_factory, dataloader_factory):
    metric = Accuracy(transform="binary_cross_entropy_with_logits_transform")

    # simulate cross entropy logits and labels
    train_data = torch.tensor([
        [1, -8, 2, 3],
        [3, -1, 9, 6],
        [-1, 9, 2, -2],
    ])
    train_label = torch.tensor([
        [True, False, True, True],
        [True, True, True, True],
        [True, False, True, False],
    ]) # acc should be 9 / 12 = 0.75
    train_dataloader = dataloader_factory(train_data, train_label, batch_size=3)

    val_data = torch.tensor([1, -1, 1, -1, 1])
    val_label = torch.tensor([True, False, True, False, False])  # acc should be : 0.8
    val_dataloader = dataloader_factory(val_data, val_label, batch_size=3)

    miner = miner_factory(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        plugins=[metric],
    )
    miner.train()

    assert metric.last_score == 0.8
    assert metric.best_score == 0.8
    assert metric.chart.points["train_acc"][0] == 0.75
    assert metric.chart.points["val_acc"][0] == 0.8
