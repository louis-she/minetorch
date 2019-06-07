import minetorch.runtime.process_env as env
from minetorch.core import Plugin
import minetorch.constants as C


class CorePlugin(Plugin):
    """The Minetorch Trainer can be runned independently.
    This plugin activate Trainer with the ability to communicate with the
    Minetorch Server with some basic data collection such as loss.
    """
    def after_init(self, payload, trainer):
        ratio = len(trainer.train_dataloader)
        env.rpc.set_timer(trainer.current_epoch, C.TIMER_EPOCH, ratio)
        env.rpc.set_timer(trainer.current_iteration, C.TIMER_ITERATION)

        env.rpc.create_graph('train_epoch_loss', C.TIMER_EPOCH)
        env.rpc.create_graph('val_epoch_loss', C.TIMER_EPOCH)
        env.rpc.create_graph('train_iteration_loss', C.TIMER_ITERATION)

    def before_epoch_start(self, payload, trainer):
        env.rpc.set_timer(trainer.current_epoch, C.TIMER_EPOCH)
        env.rpc.set_timer(trainer.current_iteration, C.TIMER_ITERATION)

    def before_train_iteration_start(self, payload, trainer):
        env.rpc.set_timer(trainer.current_iteration, C.TIMER_ITERATION)

    def after_epoch_end(self, payload, trainer):
        env.rpc.add_point('train_epoch_loss', payload['train_loss'])
        env.rpc.add_point('val_epoch_loss', payload['val_loss'])

    def after_train_iteration_end(self, payload, trainer):
        env.rpc.add_point('train_iteration_loss', payload['loss'])


class TestLoggerPlugin(Plugin):
    """This is just for dev test"""
    def before_epoch_start(self, payload, trainer):
        env.logger.error('this is a error')
        env.logger.debug('this is a debug')
        env.logger.info('this is a info')
        env.logger.warn('this is a warning')
