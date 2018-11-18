import logging
import os

import torch

from . import drawers


class Trainer(object):
    """The heart of minetorch

    Args:
        alchemistic_directory (string):
            The directory which minetorch will use to store everything in
        model (torch.nn.Module):
            Pytorch model optimizer (torch.optim.Optimizer): Pytorch optimizer
        loss_func (function):
            A special hook function to compute loss, the function receive 2 variable:
            * Trainer: the trainer object
            * Data: Batch data been yield by the loader
            return value of the hook function should be a float number of the loss
        code (str, optional):
            Defaults to "geass". It's a code name of one
            attempt. Assume one is doing kaggle competition and will try
            different models, parameters, optimizers... To keep results of every
            attempt, one should change the code name before tweaking things.
        train_dataloader (torch.utils.data.DataLoader):
            Pytorch dataloader
        val_dataloader (torch.utils.data.DataLoader, optional):
            Defaults to None, if no validation dataloader is provided, will skip validation
        resume (bool, optional):
            Defaults to True. Resume from last training, could be:
            * True: resume from the very last epochs
            * String: resume from the specified epochs
                          etc. `34`, `68` `best`
        eval_stride (int, optional):
            Defaults to 1. Validate every `eval_stride` epochs
        persist_stride (int, optional):
            Defaults to 1.
            Save model every `persist_stride` epochs
        drawer (minetorch.Drawer or string, optional):
            Defaults to None.
            If provide, Trainer will draw training loss and validation loss
            curves, could be `tensorboard` or self implemented Drawer object
        hooks (dict, optional):
            Defaults to {}. Define hook functions.
        max_epochs ([type], optional):
            Defaults to None. How many epochs to train, None means unlimited.
        logging_format ([type], optional):
            Defaults to None. logging format
    """

    def __init__(self, alchemistic_directory, model, optimizer, loss_func,
                 code="geass", train_dataloader=None, val_dataloader=None,
                 resume=True, eval_stride=1, persist_stride=1,
                 drawer=None, hooks={}, max_epochs=None,
                 logging_format=None):
        self.alchemistic_directory = alchemistic_directory
        self.code = code
        self.create_dirs()
        self.set_logging_config(alchemistic_directory, code, logging_format)
        self.create_drawer(drawer)
        self.models_dir = os.path.join(alchemistic_directory, code, 'models')

        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.loss_func = loss_func
        self.resume = resume
        self.eval_stride = eval_stride
        self.persist_stride = persist_stride

        self.lowest_train_loss = float('inf')
        self.lowest_val_loss = float('inf')
        self.current_epoch = 0
        self.hook_funcs = hooks
        self.max_epochs = max_epochs

        self.init_model()
        self.call_hook_func('after_init')
        self.status = 'init'

    def set_logging_config(self, alchemistic_directory, code, logging_format):
        self.log_dir = os.path.join(alchemistic_directory, code)
        log_file = os.path.join(self.log_dir, 'log.txt')
        logging_format = logging_format if logging_format is not None else \
            '%(levelname)s %(asctime)s %(message)s'
        logging.basicConfig(
            filename=log_file,
            format=logging_format,
            datefmt="%m-%d %H:%M:%S",
            level=logging.INFO
        )

    def create_drawer(self, drawer):
        if drawer == 'tensorboard':
            self.drawer = drawers.TensorboardDrawer(
                self.alchemistic_directory, self.code)
        else:
            self.drawer = drawer

    def init_model(self):
        """resume from some checkpoint
        """
        if self.resume is True:
            # resume from the newest model
            if os.path.isfile(self.model_file_path('latest')):
                checkpoint = self.get_checkpoint('latest')
            else:
                checkpoint = None
                logging.warning('Could not find checkpoint to resume, '
                                'train from scratch')
        elif isinstance(self.resume, str):
            checkpoint = self.get_checkpoint(self.resume)
        else:
            checkpoint = False

        if checkpoint:
            logging.info("Start to load checkpoint")
            self.current_epoch = checkpoint['epoch']
            self.lowest_train_loss = checkpoint['lowest_train_loss']
            self.lowest_val_loss = checkpoint['lowest_val_loss']

            try:
                self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            except:
                logging.warning(
                    'load checkpoint failed, the state in the '
                    'checkpoint is not matched with the model, '
                    'try to reload checkpoint with unstrict mode')
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if (self.drawer is not None) and ('drawer_state' in checkpoint):
                self.drawer.set_state(checkpoint['drawer_state'])
            logging.info('Checkpoint loaded')

    def call_hook_func(self, name):
        if name not in self.hook_funcs:
            return
        self.hook_funcs[name](self)

    def train(self):
        """start to train the model
        """
        while True:
            self.call_hook_func('before_epoch_start')
            self.current_epoch += 1

            self.model.train()
            train_iters = len(self.train_dataloader)
            val_iters = len(self.val_dataloader)

            total_train_loss = 0
            for index, data in enumerate(self.train_dataloader):
                total_train_loss += self.run_train_iteration(index, data, train_iters)
            train_loss = total_train_loss / train_iters

            total_val_loss = 0
            if self.val_dataloader is not None:
                val_iters = len(self.val_dataloader)
                with torch.set_grad_enabled(False):
                    self.model.eval()
                    for index, data in enumerate(self.val_dataloader):
                        total_val_loss += self.run_val_iteration(index, data, val_iters)
                val_loss = total_val_loss / val_iters

            if self.drawer is not None:
                self.drawer.scalars(
                    {'train': train_loss, 'val': val_loss}, 'loss'
                )

            if train_loss < self.lowest_train_loss:
                self.lowest_train_loss = train_loss

            if val_loss < self.lowest_val_loss:
                logging.info(
                    'current val loss {} is lower than lowest {}, '
                    'persist this model as best one'.format(
                        val_loss, self.lowest_val_loss))

                self.lowest_val_loss = val_loss
                self.persist('best')
            self.persist('latest')

            if not self.current_epoch % self.persist_stride:
                self.persist('epoch_{}'.format(self.current_epoch))

            self.call_hook_func('after_epoch_end')

            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.call_hook_func('before_quit')
                logging.info('exceed max epochs, quit!')
                break

    def run_train_iteration(self, index, data, train_iters):
        self.status = 'train'
        self.call_hook_func('before_train_iteration_start')

        loss = self.loss_func(self, data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.detach()
        logging.info('[train {}/{}/{}] loss {}'.format(
            self.current_epoch, index, train_iters, loss))
        if loss < self.lowest_train_loss:
            self.lowest_train_loss = loss

        self.call_hook_func('after_train_iteration_end')
        return loss

    def run_val_iteration(self, index, data, val_iters):
        self.status = 'val'
        self.call_hook_func('before_val_iteration_start')
        loss = self.loss_func(self, data)
        loss = loss.detach()
        logging.info('[val {}/{}/{}] loss {}'.format(
            self.current_epoch, index, val_iters, loss))

        self.call_hook_func('after_val_iteration_ended')
        return loss

    def persist(self, name):
        """save the model to disk
        """
        self.call_hook_func('before_checkpoint_persisted')
        if self.drawer is not None:
            drawer_state = self.drawer.get_state()
        else:
            drawer_state = {}

        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'lowest_train_loss': self.lowest_train_loss,
            'lowest_val_loss': self.lowest_val_loss,
            'drawer_state': drawer_state
        }

        torch.save(state, self.model_file_path(name))
        logging.info('save checkpoint to {}'.format(self.model_file_path(name)))
        self.call_hook_func('after_checkpoint_persisted')


    def model_file_path(self, model_name):
        model_file_name = '{}.pth.tar'.format(model_name)
        return os.path.join(self.models_dir, model_file_name)

    def get_checkpoint(self, name):
        return torch.load(self.model_file_path(name))

    # TODO: implement methods below
    def graceful_stop(self):
        """stop train and exist after this epoch
        """
        pass

    def save_and_stop(self):
        """save the model immediately and stop training
        """
        pass

    def create_dirs(self):
        """Create directories
        """
        self.create_dir('')
        self.create_dir(self.code)
        self.create_dir(self.code, 'models')

    def create_dir(self, *args):
        """Create directory
        """
        current_dir = self.alchemistic_directory
        for dir_name in args:
            current_dir = os.path.join(current_dir, dir_name)
            if not os.path.isdir(current_dir):
                os.mkdir(current_dir)
