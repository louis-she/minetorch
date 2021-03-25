from ignite.engine.events import Events
from minetorch.core.plugin import Plugin, on

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Please install `tqdm` before use the pbar"
    )


class Pbar(Plugin):

    def __init__(self, log_interval: int = 10):
        super().__init__()
        self.log_interval = log_interval

    @on(Events.ITERATION_COMPLETED, lambda self: dict(every=self.log_interval))
    def _update_pbar(self):
        self.pbar.desc = f"epoch: {self.engine.state.epoch} - loss: {self.engine.state.output:.2f}"
        self.pbar.update(self.log_interval)

    @on(Events.STARTED)
    def _create_pbar(self):
        self.pbar = tqdm(
            initial=0,
            leave=False,
            total=self.engine.state.epoch_length,
            desc=f"epoch: {self.engine.state.epoch} - loss: {0:.2f}")

    @on(Events.COMPLETED)
    def _close_pbar(self):
        self.pbar.close()

    @on(Events.EPOCH_COMPLETED)
    def _refresh_pbar(self):
        self.pbar.refresh()
        self.pbar.n = self.pbar.last_print_n = 0
