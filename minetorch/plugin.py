import os
from pathlib import Path


class Plugin():

    def __init__(self):
        self.name = self.__class__.__name__
        self.miner = None

    def before_hook(self, hook_name, payload):
        return True

    def set_miner(self, miner):
        self.miner = miner

    def notify(self, message, _type='info'):
        message = f"[{self.name}] {message}"
        self.miner.notify(message, _type)

    def __getattr__(self, key):
        if self.miner is None or not hasattr(self.miner, key):
            raise AttributeError(key)
        return getattr(self.miner, key)

    def print_txt(self, printable, name):
        print_file = Path(self.alchemistic_directory) / self.code / 'prints' / name
        print_file.parent.mkdir(parents=True, exist_ok=True)
        with open(print_file, 'a') as f:
            print(f'================ Epoch {self.current_epoch} ================\n', file=f)
            print(printable, file=f)
            print("\n\n", file=f)
