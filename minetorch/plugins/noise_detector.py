import torch
from minetorch.plugin import Plugin
from minetorch.statable import Statable

class NoiseSampleDetector(Plugin, Statable):
    """This plugin helps to find out the suspicious noise samples.
    provid a metric which compute a scalar for every sample, in most cases
    the metric should be the loss function without reduce.
    """

    def __init__(self, metric, topn=50):
        super().__init__()
        self.metric = metric
        self.topn = topn
        self.train_metrics = []
        self.val_metrics = []

    def before_init(self):
        self.miner.statable[self.__class__.__name__] = self
        self.train_dataloader = torch.utils.data.DataLoader(
            self.miner.train_dataloader.dataset,
            batch_size=self.miner.train_dataloader.batch_size,
            num_workers=self.miner.train_dataloader.num_workers,
            shuffle=False
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.miner.val_dataloader.dataset,
            batch_size=self.miner.train_dataloader.batch_size,
            num_workers=self.miner.train_dataloader.num_workers,
            shuffle=False
        )

    def load_state_dict(self, data):
        self.train_metrics = data[0]
        self.val_metrics = data[1]

    def state_dict(self):
        return (self.train_metrics, self.val_metrics)

    def after_epoch_end(self, **kwargs):
        with torch.no_grad():
            self.train_metrics.append(
                self._predict_dataset(self.train_dataloader)
            )
            self.val_metrics.append(
                self._predict_dataset(self.val_dataloader)
            )

        _, train_indices = torch.sort(torch.std(torch.stack(self.train_metrics), dim=0), descending=True)
        _, val_indices = torch.sort(torch.std(torch.stack(self.val_metrics), dim=0), descending=True)

        self.print_txt(f"Train dataset most {self.topn} suspicious indices: {train_indices.tolist()[:self.topn]} \n"
                       f"Validation dataset most {self.topn} suspicious indices: {val_indices.tolist()[:self.topn]}",
                       'suspicious_noise_samples')

    def _predict_dataset(self, dataloader):
        results = torch.zeros([len(dataloader.dataset)])
        for index, data in enumerate(dataloader):
            predict = self.model(data[0].to(self.devices))
            offset = index * dataloader.batch_size
            results[offset:offset + dataloader.batch_size] = self.metric(predict, data[1].to(self.devices)).detach().cpu()
        return results
