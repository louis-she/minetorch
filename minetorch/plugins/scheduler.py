from minetorch.plugin import Plugin
from minetorch.statable import Statable


class Scheduler(Plugin, Statable):

    def __init__(self, scheduler, iter_step=False):
        super().__init__()
        self.torch_scheduler = scheduler
        self.iter_step = iter_step
        self.steps = 0
        self.lrs = []

    def load_state_dict(self, state):
        self.lrs = state["lrs"]
        self.steps = state["steps"]
        self.torch_scheduler.load_state_dict(state["torch_scheduler"])

    def state_dict(self):
        return {
            "lrs": self.lrs,
            "steps": self.steps,
            "torch_scheduler": self.torch_scheduler.state_dict()
        }

    def step(self):
        self.torch_scheduler.step()
        self.lrs.append(self.torch_scheduler.get_last_lr())
        self.stpes += 1

    def after_epoch_end(self):
        if not self.iter_step:
            self.step()
            self.scalars(self.steps, )

    def after_train_iteration_end(self):
        if self.iter_step:
            self.step()
