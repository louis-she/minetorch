import logging
import os
import time
from pathlib import Path

import functools
import torch
import tqdm
from IPython.core.display import HTML, display
import numpy as np
from . import drawers


class Miner(object):
    """The heart of minetorch

    Args:
        alchemistic_directory (string):
            The directory which minetorch will use to store everything in
        model (torch.nn.Module):
            Pytorch model optimizer (torch.optim.Optimizer): Pytorch optimizer
        loss_func (function):
            A special hook function to compute loss, the function receive 2 variable:
            * Miner: the miner object
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
            Defaults to matplotlib.
            If provide, Miner will draw training loss and validation loss
            curves, could be `tensorboard` or self implemented Drawer object
        hooks (dict, optional):
            Defaults to {}. Define hook functions.
        max_epochs ([type], optional):
            Defaults to None. How many epochs to train, None means unlimited.
        logging_format ([type], optional):
            Defaults to None. logging format
        trival ([Boolean], optional):
            Defaults to False. If true, both training and validation
            process will be breaked in 10 iterations
        plugins (list, optional):
            Defaults to []. This is actually a collection of `hooks`, do not set
            hooks and plugins the same time.
    """

    def __init__(self, alchemistic_directory, model, optimizer, loss_func, metrics=[],
                 code="geass", train_dataloader=None, val_dataloader=None,
                 resume=True, eval_stride=1, persist_stride=1, gpu=True,
                 drawer='matplotlib', hooks={}, max_epochs=None, statable={},
                 logging_format=None, trival=False, in_notebook=False, plugins=[],
                 logger=None):

        self.alchemistic_directory = alchemistic_directory
        self.code = code
        self.create_dirs()
        self.gpu = gpu
        self.devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        if self.logger is None:
            self.set_logging_config(alchemistic_directory, code, logging_format)
            self.logger = logging
        self.create_drawer(drawer)
        self.models_dir = os.path.join(alchemistic_directory, code, 'models')
        self.in_notebook = in_notebook
        self.statable = statable

        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.loss_func = loss_func
        self.metrics = metrics
        for metric in self.metrics:
            if not hasattr(metric, 'keywords'):
                metric = functools.partial(metric, func=lambda x: x, separate_class=True)

        self.resume = resume
        self.eval_stride = eval_stride
        self.persist_stride = persist_stride

        self.lowest_train_loss = float('inf')
        self.lowest_val_loss = float('inf')
        self.current_epoch = 0
        self.current_train_iteration = 0
        self.current_val_iteration = 0
        self.hook_funcs = hooks
        self.max_epochs = max_epochs
        self.metrics_metadata = {}

        self.init_model()
        self.trival = trival

        self._set_tqdm()
        self._check_statable()

        self.plugins = plugins
        for plugin in self.plugins:
            plugin.set_miner(self)

        self.status = 'init'
        self.call_hook_func('after_init')

    def _check_statable(self):
        for name, statable in self.statable.items():
            if not (hasattr(statable, 'state_dict') and hasattr(statable, 'load_state_dict')):
                raise Exception(f'The {name} is not a statable object')

    def _set_tqdm(self):
        if self.in_notebook:
            self.tqdm = tqdm.tqdm_notebook
        else:
            self.tqdm = lambda x: x

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
        elif drawer == 'matplotlib':
            self.drawer = drawers.MatplotlibDrawer(
                self.alchemistic_directory, self.code)
        else:
            self.drawer = drawer

    def notebook_output(self, message, _type='info'):
        type_config = {
            'info': ['💬', '#6f818a'],
            'success': ['✅', '#7cb305'],
            'error': ['❌', '#cf1322'],
            'warning': ['⚠️', '#d46b08'],
        }[_type]
        if self.in_notebook:
            display(HTML(
                f'<div style="font-size: 12px; color: {type_config[1]}">'
                f'⏰ {time.strftime("%b %d - %H:%M:%S")} >>> '
                f'{type_config[0]} {message}'
                '</div>'
            ))

    def notebook_divide(self, message):
        if self.in_notebook:
            display(HTML(
                '<div style="display: flex; justify-content: center;">'
                f'<h3 style="color: #7cb305; border-bottom: 4px dashed #91d5ff; padding-bottom: 6px;">{message}</h3>'
                '</div>'
            ))

    def init_model(self):
        """resume from some checkpoint
        """
        if isinstance(self.model, torch.nn.DataParallel):
            raise Exception(
                'Don\'t parallel the model yourself, instead, if the '
                '`gpu` option is true(default), Minetorch will do this for you.'
            )

        if self.resume is True:
            # resume from the newest model
            if self.model_file_path('latest') is not None:
                checkpoint_path = self.model_file_path('latest')
            else:
                checkpoint_path = None
                msg = ('Could not find checkpoint to resume, '
                       'train from scratch')
                self.logger.warning(msg)
                self.notebook_output(msg, _type='warning')
        elif isinstance(self.resume, str):
            checkpoint_path = self.model_file_path(self.resume)
        elif isinstance(self.resume, int):
            checkpoint_path = self.model_file_path(str(self.resume))
        else:
            checkpoint_path = None

        if self.resume is not True and self.resume and checkpoint_path is None:
            # user has specified a none existed model, should raise a error
            raise Exception(f"Could not find model {self.resume}")

        if checkpoint_path is not None:
            msg = f"Start to load checkpoint {checkpoint_path}"
            self.logger.info(msg)
            self.notebook_output(msg)
            checkpoint = torch.load(checkpoint_path)
            self.current_epoch = checkpoint['epoch']
            self.current_train_iteration = checkpoint['train_iteration']
            self.current_val_iteration = checkpoint['val_iteration']
            self.lowest_train_loss = checkpoint['lowest_train_loss']
            self.lowest_val_loss = checkpoint['lowest_val_loss']

            try:
                self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            except:
                msg = ('load checkpoint failed, the state in the '
                       'checkpoint is not matched with the model, '
                       'try to reload checkpoint with unstrict mode')
                self.logger.warning(msg)
                self.notebook_output(msg)
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)

            if 'optimizer' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                except:
                    msg = ('load optimizer state failed, will skip this error and continue, '
                            'stop the process if it is not expected')
                    self.logger.warning(msg)
                    self.notebook_output(msg)

            if (self.drawer is not None) and ('drawer_state' in checkpoint):
                self.drawer.set_state(checkpoint['drawer_state'])

            if 'statable' in checkpoint:
                for name, statable in self.statable.items():
                    if name not in checkpoint['statable']:
                        continue
                    statable.load_state_dict(checkpoint['statable'][name])
            msg = 'checkpoint loaded'
            self.notebook_output(msg, _type='success')

        if self.gpu:
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                self.notify('no GPU detected, will train on CPU.')
            else:
                self.notify(f'found {gpu_count} GPUs, will use all of them to train')
                devices = list(map(lambda x: f'cuda:{x}', range(gpu_count)))
                self.model.cuda()
                self.model = torch.nn.DataParallel(self.model, devices)

    def notify(self, message, _type='info'):
        getattr(self.logger, _type)(message)
        self.notebook_output(message, _type)

    def call_hook_func(self, name, **payload):
        if name in self.hook_funcs:
            self.hook_funcs[name](payload)
        else:
            for plugin in self.plugins:
                if not plugin.before_hook(name, payload):
                    continue
                if hasattr(plugin, name):
                    getattr(plugin, name)(payload)

    def train(self):
        """start to train the model
        """
        while True:
            self.current_epoch += 1
            self.call_hook_func('before_epoch_start', epoch=self.current_epoch)
            self.notebook_divide(f'Epoch {self.current_epoch}')
            self.model.train()
            train_iters = len(self.train_dataloader)

            total_train_loss = 0
            total_train_metrics = {}
            self.notebook_output(f'start to train epoch {self.current_epoch}')
            for index, data in enumerate(self.tqdm(self.train_dataloader)):
                if self.trival is True and index == 10:
                    break
                train_loss, train_metrics = self.run_train_iteration(index, data, train_iters)
                total_train_loss += train_loss
                for metric in train_metrics:
                    if metric not in total_train_metrics:
                        total_train_metrics[metric] = 0
                    total_train_metrics[metric] += train_metrics[metric]
            for i, j in enumerate(total_train_metrics):
                total_train_metrics[j] = total_train_metrics[j] / train_iters
                total_train_metrics[j] = self.metrics[i].keywords['func'](total_train_metrics[j])

            total_train_loss = total_train_loss / train_iters
            self.notebook_output(f'training of epoch {self.current_epoch} finished, '
                                 f'loss is {total_train_loss}')

            total_val_loss = 0
            total_val_metrics = {}
            if self.val_dataloader is not None:
                val_iters = len(self.val_dataloader)
                with torch.set_grad_enabled(False):
                    self.model.eval()
                    self.notebook_output(f'validate epoch {self.current_epoch}')
                    for index, data in enumerate(self.tqdm(self.val_dataloader)):
                        if self.trival is True and index == 10:
                            break
                        val_loss, val_metrics = self.run_val_iteration(index, data, val_iters)
                        total_val_loss += val_loss
                        for metric in val_metrics:
                            if metric not in total_val_metrics.keys():
                                total_val_metrics[metric] = 0
                                total_val_metrics[metric] += val_metrics[metric]
                            else:
                                total_val_metrics[metric] += val_metrics[metric]
                for i, j in enumerate(total_val_metrics):
                    total_val_metrics[j] = total_val_metrics[j] / val_iters
                    total_val_metrics[j] = self.metrics[i].keywords['func'](total_val_metrics[j])

                total_val_loss = total_val_loss / val_iters
                self.notebook_output(f'validation of epoch {self.current_epoch}'
                                     f'finished, loss is {total_val_loss}')
            self.call_hook_func(
                'after_epoch_end',
                train_loss=total_train_loss,
                val_loss=total_val_loss,
                train_metric=train_metrics,
                val_metric=val_metrics,
                epoch=self.current_epoch
                )

            if self.drawer is not None:
                self.drawer.scalars(
                    {'train': total_train_loss, 'val': total_val_loss}, 'loss'
                )
                for metric in self.metrics:
                    if total_train_metrics[metric.keywords['func'].__name__].shape.numel() != 1:
                        for i in range(total_train_metrics[metric.keywords['func'].__name__].shape.numel()):
                            self.drawer.scalars(
                                {
                                    'train': total_train_metrics[metric.keywords['func'].__name__][i],
                                    'val': total_val_metrics[metric.keywords['func'].__name__][i]
                                    },
                                metric.keywords['func'].__name__+'_class_{}'.format(i+1)
                                )
                    else:
                        self.drawer.scalars(
                            {'train': total_train_metrics[metric.keywords['func'].__name__],
                            'val': total_val_metrics[metric.keywords['func'].__name__]},
                            metric.keywords['func'].__name__
                            )

            if total_train_loss < self.lowest_train_loss:
                self.lowest_train_loss = total_train_loss

            if total_val_loss < self.lowest_val_loss:
                message = ('current val loss {} is lower than lowest {}, '
                           'persist this model as best one'.format(
                            total_val_loss, self.lowest_val_loss))
                self.notebook_output(f'{message}', _type='success')
                self.logger.info(message)

                self.lowest_val_loss = total_val_loss
                self.persist('best')
            self.persist('latest')

            if not self.current_epoch % self.persist_stride:
                self.persist('epoch_{}'.format(self.current_epoch))

            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.call_hook_func('before_quit')
                self.logger.info('exceed max epochs, quit!')
                break

    def run_train_iteration(self, index, data, train_iters):
        self.status = 'train'
        self.current_train_iteration += 1
        self.call_hook_func(
            'before_train_iteration_start',
            data=data,
            index=index,
            total_iters=train_iters,
            iteration=self.current_train_iteration
            )
        batch_size = data[0].shape[0]
        predict = self.model(data[0].to(self.devices))
        loss = self.loss_func(predict, data[1].to(self.devices))
        train_metrics = {}
        for metric in self.metrics: # iterate metrics specified by users
            train_metrics[metric.keywords['func'].__name__] = 0
            for batch in range(batch_size): # iterate data from the dataloader
                values, _ = metric(predict[batch].detach().cpu(), data[1][batch]) # return confusion_matrix and specific functions
                train_metrics[metric.keywords['func'].__name__] += values  # predict.shape = [B,C,H,W]
            train_metrics[metric.keywords['func'].__name__] /= batch_size

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.detach().cpu().item()
        self.logger.info('[train {}/{}/{}] loss {}'.format(
            self.current_epoch, index, train_iters, loss))
        if loss < self.lowest_train_loss:
            self.lowest_train_loss = loss

        self.call_hook_func(
            'after_train_iteration_end',
            loss=loss,
            data=data,
            index=index,
            total_iters=train_iters,
            iteration=self.current_train_iteration
            )
        return loss, train_metrics

    def run_val_iteration(self, index, data, val_iters):
        self.status = 'val'
        self.current_val_iteration += 1
        self.call_hook_func(
            'before_val_iteration_start',
            data=data,
            index=index,
            total_iters=val_iters,
            iteration=self.current_val_iteration
            )
        batch_size = data[0].shape[0]
        predict = self.model(data[0].to(self.devices))
        loss = self.loss_func(predict, data[1].to(self.devices))
        val_metrics = {}
        for metric in self.metrics:
            val_metrics[metric.keywords['func'].__name__] = 0
            for batch in range(batch_size):
                values, func = metric(predict[batch].detach().cpu(), data[1][batch])
                val_metrics[metric.keywords['func'].__name__] += values  # predict.shape = [B,C,H,W]
            val_metrics[metric.keywords['func'].__name__] /= batch_size
        loss = loss.detach().cpu().item()
        self.logger.info('[val {}/{}/{}] loss {}'.format(
            self.current_epoch, index, val_iters, loss))

        self.call_hook_func(
            'after_val_iteration_ended',
            predicts=predict,
            loss=loss,
            data=data,
            index=index,
            total_iters=val_iters,
            iteration=self.current_val_iteration
            )
        return loss, val_metrics

    def persist(self, name):
        """save the model to disk
        """
        self.call_hook_func('before_checkpoint_persisted')
        if self.drawer is not None:
            drawer_state = self.drawer.get_state()
        else:
            drawer_state = {}

        if isinstance(self.model, torch.nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'train_iteration': self.current_train_iteration,
            'val_iteration': self.current_val_iteration,
            'lowest_train_loss': self.lowest_train_loss,
            'lowest_val_loss': self.lowest_val_loss,
            'drawer_state': drawer_state,
            'statable': {}
        }

        for statable_name, statable in self.statable.items():
            state['statable'][statable_name] = statable.state_dict()

        modelpath = self.standard_model_path(name)
        torch.save(state, modelpath)
        message = f'save checkpoint to {self.standard_model_path(name)}'
        self.logger.info(message)
        self.notebook_output(message)
        self.call_hook_func('after_checkpoint_persisted', modelpath=modelpath)

    def standard_model_path(self, model_name):
        return os.path.join(self.models_dir, f'{model_name}.pth.tar')

    def model_file_path(self, model_name):
        model_name_path = Path(model_name)
        models_dir_path = Path(self.models_dir)

        search_paths = [
            model_name_path,
            models_dir_path / model_name_path,
            models_dir_path / f'{model_name}.pth.tar',
            models_dir_path / f'epoch_{model_name}.pth.tar',
        ]

        for path in search_paths:
            if path.is_file():
                return path.resolve()

        return None

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
