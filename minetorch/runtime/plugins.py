import minetorch.runtime.process_env as env
from minetorch.core import Plugin
import minetorch.constants as C


class CorePlugin(Plugin):
    """The Minetorch Trainer can be runned independently.
    This plugin activate Trainer with the ability to communicate with the
    Minetorch Server with some basic data collection such as loss.
    """
    def before_epoch_start(self, payload, trainer):
        env.rpc.set_timer(trainer.current_epoch, C.TIMER_EPOCH)
        env.rpc.set_timer(trainer.current_iteration, C.TIMER_ITERATION)
        env.rpc.set_timer(None, C.TIMER_SNAPSHOT)

        env.rpc.create_graph('train_epoch_loss', C.TIMER_EPOCH)
        env.rpc.create_graph('val_epoch_loss', C.TIMER_EPOCH)
        env.rpc.create_graph('train_iteration_loss', C.TIMER_ITERATION)

    def before_train_iteration_start(self, payload, trainer):
        env.rpc.set_timer(trainer.current_iteration, C.TIMER_ITERATION)

    def after_epoch_end(self, payload, trainer):
        env.rpc.add_point('train_epoch_loss', payload['train_loss'])
        env.rpc.add_point('val_epoch_loss', payload['val_loss'])

    def after_train_iteration_end(self, payload, trainer):
        env.rpc.add_point('train_iteration_loss', payload['loss'])
