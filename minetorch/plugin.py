import os


class Plugin:
    def __init__(self, prefix=""):
        self.name = self.__class__.__name__
        self.miner = None
        self.prefix = prefix

    def before_hook(self, hook_name, payload):
        return True

    def set_miner(self, miner):
        self.miner = miner

    def notify(self, message, _type="info"):
        message = f"[{self.name}] {message}"
        self.miner.notify(message, _type)

    def __getattr__(self, key):
        if self.miner is None or not hasattr(self.miner, key):
            raise AttributeError(key)
        return getattr(self.miner, key)

    def print_txt(self, printable, name):
        with open(self.plugin_file(f"{name}.txt"), "a") as f:
            print(
                f"================ Epoch {self.current_epoch} ================\n",
                file=f,
            )
            print(printable, file=f)
            print("\n\n", file=f)

    @property
    def plugin_dir(self):
        if hasattr(self, "_plugin_dir"):
            return getattr(self, "_plugin_dir")

        plugin_dir = os.path.join(self.code_dir, self.__class__.__name__)
        try:
            os.mkdir(plugin_dir)
        except FileExistsError:
            pass
        self._plugin_dir = plugin_dir
        return self._plugin_dir

    def plugin_file(self, name):
        return os.path.join(self.plugin_dir, name)

    def create_sheet_column(self, key, name):
        self.miner.create_sheet_column(f"{self.prefix}{key}", f"{self.prefix}{name}")

    def update_sheet(self, key, value):
        self.miner.update_sheet(f"{self.prefix}{key}", value)

    def scalars(self, values, graph):
        return self.miner.drawer.scalars(
            self.miner.current_epoch, values, f"{self.prefix}{graph}"
        )
