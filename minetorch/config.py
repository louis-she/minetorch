import os
import yaml
from dataclasses import dataclass
from minetorch.plugin import Plugin
from torch.utils.tensorboard import SummaryWriter


class ConfigBase(Plugin):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._c = {}

    def __getattr__(self, key):
        return self._c.get(key, None)

    def after_init(self):
        super().after_init()
        tensorboard_dir = os.path.join(self.miner.code_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        self.writer.add_text(
            "config",
            yaml.dump(self._c),
            global_step=self.miner.current_epoch
        )

    @classmethod
    def load(cls, name):
        with open("./config.yml", "r") as f:
            data = yaml.load(f, yaml.Loader)
        if name not in data:
            raise RuntimeError(f"config {name} is not exists")
        obj = cls()
        for key, value in data[name].items():
            obj._c[key] = value
        return obj

    def load_next(self):
        pass
