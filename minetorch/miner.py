import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Union

import torch
import tqdm
from IPython.core.display import HTML, display
from minetorch.charts import Chart, TensorBoardChart

from minetorch.plugin import Plugin


class Miner(object):
    """The heart of minetorch

    Args:
        base_dir (string):
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
        chart (minetorch.charts.Chart, optional):
            Defaults to minetorch.charts.TensorBoardChart.
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
        forward (function, optional):
            custom forward function.
        verbose (boolean, optional):
            los loss of every iteration
    """

    def __init__(
        self,
        base_dir: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_func: Callable,
        code: str = "geass",
        train_dataloader: Iterable = None,
        val_dataloader: Iterable = None,
        resume: Union[int, str, bool] = True,
        eval_stride: int = 1,
        persist_stride: int = 1,
        gpu: bool = True,
        chart_type: Chart = TensorBoardChart,
        hooks={},
        max_epochs: int = None,
        logging_format: str = None,
        trival: bool = False,
        in_notebook: bool = False,
        plugins: List[Plugin] = [],
        logger: logging.Logger = None,
        accumulated_iter: int = 1,
        ignore_optimizer_resume: bool = False,
        forward: Callable = None,
        verbose: bool = False,
        amp: bool = False,
        amp_scaler: bool = True,
    ):
        self.base_dir = base_dir
        self.code = code
        if trival:
            self.code = f"trival_{code}"
        self.create_dirs()
        self.gpu = gpu
        self.devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.code_dir = os.path.join(base_dir, self.code)
        if self.logger is None:
            self.set_logging_config(base_dir, self.code, logging_format)
            self.logger = logging
        self.chart_type = chart_type
        self.models_dir = os.path.join(base_dir, self.code, "models")
        self.in_notebook = in_notebook
        self.accumulated_iter = float(accumulated_iter)
        self.ignore_optimizer_resume = ignore_optimizer_resume

        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.loss_func = loss_func

        self.resume = resume
        self.eval_stride = eval_stride
        self.persist_stride = persist_stride
        self.lowest_train_loss = float("inf")
        self.lowest_val_loss = float("inf")
        self.current_epoch = 0
        self.current_train_iteration = 0
        self.current_val_iteration = 0
        self.hook_funcs = hooks
        self.max_epochs = max_epochs
        self.trival = trival
        self.forward_fn = forward
        self.verbose = verbose
        self.amp = amp
        self.amp_scaler = amp_scaler

        if self.amp and self.amp_scaler:
            self.scaler = torch.cuda.amp.GradScaler()

        self.plugins = plugins
        for plugin in self.plugins:
            plugin.set_miner(self)

        self._set_tqdm()
        self.call_hook_func("before_init")
        self.init_model()
        self.loss_chart = self.create_chart("loss")
        self.status = "init"
        self.call_hook_func("after_init")

    def _set_tqdm(self):
        if self.in_notebook:
            self.tqdm = tqdm.notebook.tqdm
        else:
            self.tqdm = tqdm.tqdm

    def create_chart(self, name: str):
        return self.chart_type(self, name)

    def set_logging_config(self, base_dir, code, logging_format):
        self.log_dir = os.path.join(base_dir, code)
        log_file = os.path.join(self.log_dir, "log.txt")
        logging_format = (
            logging_format
            if logging_format is not None
            else "%(levelname)s %(asctime)s %(message)s"
        )
        logging.basicConfig(
            filename=log_file,
            format=logging_format,
            datefmt="%m-%d %H:%M:%S",
            level=logging.INFO,
        )

    def notebook_output(self, message, _type="info"):
        type_config = {
            "info": ["💬", "#6f818a"],
            "success": ["✅", "#7cb305"],
            "error": ["❌", "#cf1322"],
            "warning": ["⚠️", "#d46b08"],
        }[_type]
        if self.in_notebook:
            display(
                HTML(
                    f'<div style="font-size: 12px; color: {type_config[1]}">'
                    f'⏰ {time.strftime("%b %d - %H:%M:%S")} >>> '
                    f"{type_config[0]} {message}"
                    "</div>"
                )
            )

    def notebook_divide(self, message):
        if self.in_notebook:
            display(
                HTML(
                    '<div style="display: flex; justify-content: center;">'
                    f'<h3 style="color: #7cb305; border-bottom: 4px dashed #91d5ff; padding-bottom: 6px;">{message}</h3>'
                    "</div>"
                )
            )

    def init_model(self):
        """resume from some checkpoint"""
        if isinstance(self.model, torch.nn.DataParallel):
            raise Exception(
                "Don't parallel the model yourself, instead, if the "
                "`gpu` option is true(default), MineTorch will do this for you."
            )

        if self.resume is True:
            # resume from the newest model
            if self.model_file_path("latest") is not None:
                checkpoint_path = self.model_file_path("latest")
            else:
                checkpoint_path = None
                msg = "Could not find checkpoint to resume, train from scratch"
                self.notify(msg, "warning")
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
            self.notify(msg)
            checkpoint = torch.load(checkpoint_path)
            self.current_epoch = checkpoint.get("epoch", 0)
            self.current_train_iteration = checkpoint.get("train_iteration", 0)
            self.current_val_iteration = checkpoint.get("val_iteration", 0)
            self.lowest_train_loss = checkpoint.get("lowest_train_loss", 9999)
            self.lowest_val_loss = checkpoint.get("lowest_val_loss", 9999)

            # load model state
            try:
                self.model.load_state_dict(checkpoint["state_dict"], strict=True)
            except Exception as e:
                msg = (
                    f"load checkpoint failed with {e}, the state in the "
                    "checkpoint is not matched with the model, "
                    "try to reload checkpoint with unstrict mode"
                )
                self.notify(msg, "warning")
                self.model.load_state_dict(checkpoint["state_dict"], strict=False)

            # load optimizer state
            if "optimizer" in checkpoint and not self.ignore_optimizer_resume:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                except Exception as e:
                    msg = (
                        f"load optimizer state failed with {e}, will skip this error and continue, "
                        "stop the process if it is not expected"
                    )
                    self.notify(msg, "warning")

            # load scaler state
            if self.amp and self.amp_scaler:
                try:
                    self.scaler.load_state_dict(checkpoint["scaler"])
                except Exception as e:
                    msg = (
                        f"load scaler state failed with {e}, will skip this error and continue, "
                        "stop the process if it is not expected"
                    )
                    self.notify(msg, "warning")

            # load plugin states
            for plugin in self.plugins:
                key = f"__plugin.{plugin.__class__.__name__}__"
                plugin.load_state_dict(checkpoint.get(key, {}))

            msg = "checkpoint loaded"
            self.notify(msg, "success")
        self.model = self.parallel_model(self.model)

    def parallel_model(self, model):
        if self.gpu:
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                self.notify("no GPU detected, will train on CPU.")
            else:
                self.notify(f"found {gpu_count} GPUs, will use all of them to train")
                devices = list(map(lambda x: f"cuda:{x}", range(gpu_count)))
                model.cuda()
                model = torch.nn.DataParallel(model, devices)
        return model

    def notify(self, message, _type="info"):
        getattr(self.logger, "info" if _type == "success" else _type)(message)
        self.notebook_output(message, _type)

    def call_hook_func(self, name, **payload):
        if name in self.hook_funcs:
            self.hook_funcs[name](miner=self, **payload)

        for plugin in self.plugins:
            if not plugin.before_handler(name, payload):
                continue
            if hasattr(plugin, name):
                getattr(plugin, name)(**payload)

    def train(self):
        """start to train the model"""
        while True:
            self.current_epoch += 1
            self.call_hook_func("before_epoch_start")
            self.notebook_divide(f"Epoch {self.current_epoch}")
            self.model.train()
            train_iters = len(self.train_dataloader)

            self.epoch_train_loss = 0
            percentage = 0
            total = len(self.train_dataloader)
            self.notify(f"start to train epoch {self.current_epoch}")
            self._update_progress(
                force=True,
                epoch=self.current_epoch,
                train_percentage="0%",
                val_percentage="0%",
            )
            t = self.tqdm(self.train_dataloader)
            for index, data in enumerate(t):
                if self.trival is True and index == 10:
                    break
                train_loss = self.run_train_iteration(index, data, train_iters)
                t.set_postfix({"train loss": train_loss})
                if int((index + 1) % self.accumulated_iter) == 0:
                    if self.amp and self.amp_scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    if self.amp and self.amp_scaler:
                        self.optimizer.zero_grad()
                    else:
                        self.optimizer.zero_grad(set_to_none=True)
                self.epoch_train_loss += train_loss
                current_percentage = math.ceil(index / total * 100)
                if current_percentage != percentage:
                    self._update_progress(train_percentage=f"{percentage}%")
                    percentage = current_percentage
            if self.amp and self.amp_scaler:
                self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad(set_to_none=True)
            self._update_progress(force=True, train_percentage=f"{current_percentage}%")

            self.epoch_train_loss = self.epoch_train_loss / train_iters
            self.notify(
                f"training of epoch {self.current_epoch} finished, "
                f"loss is {self.epoch_train_loss}"
            )

            self.epoch_val_loss = 0
            percentage = 0
            total = len(self.val_dataloader)
            if self.val_dataloader is not None:
                val_iters = len(self.val_dataloader)
                with torch.set_grad_enabled(False):
                    self.model.eval()
                    self.notify(f"validate epoch {self.current_epoch}")
                    t = self.tqdm(self.val_dataloader)
                    for index, data in enumerate(t):
                        if self.trival is True and index == 10:
                            break
                        val_loss = self.run_val_iteration(index, data, val_iters)
                        t.set_postfix({"val loss": val_loss})
                        self.epoch_val_loss += val_loss
                        current_percentage = math.ceil(index / total * 100)
                        if current_percentage != percentage:
                            self._update_progress(val_percentage=f"{percentage}%")
                            percentage = current_percentage
                    self._update_progress(
                        force=True, val_percentage=f"{current_percentage}%"
                    )

                self.epoch_val_loss = self.epoch_val_loss / val_iters
                self.notify(
                    f"validation of epoch {self.current_epoch} "
                    f"finished, loss is {self.epoch_val_loss}"
                )

            self.loss_chart.add_points(
                train=self.epoch_train_loss,
                val=self.epoch_val_loss,
            )

            if self.epoch_train_loss < self.lowest_train_loss:
                self.lowest_train_loss = self.epoch_train_loss

            should_persist_best = False
            if self.epoch_val_loss < self.lowest_val_loss:
                message = (
                    "current val loss {} is lower than lowest {}, "
                    "persist this model as best one".format(
                        self.epoch_val_loss, self.lowest_val_loss
                    )
                )
                self.notify(message, "success")
                self.lowest_val_loss = self.epoch_val_loss
                should_persist_best = True

            self.call_hook_func("before_persist_checkpoint")

            if should_persist_best:
                self.persist("best")
            self.persist("latest")
            if not self.current_epoch % self.persist_stride:
                self.persist("epoch_{}".format(self.current_epoch))

            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.call_hook_func("before_quit")
                self.notify("exceed max epochs, quit!")
                break

            self.call_hook_func("after_epoch_end")

    def run_train_iteration(self, index, data, train_iters):
        self.status = "train"
        self.current_train_iteration += 1
        self.call_hook_func(
            "before_train_iteration_start",
            data=data,
            index=index,
            total_iters=train_iters,
            iteration=self.current_train_iteration,
        )
        if self.amp and self.amp_scaler:
            with torch.cuda.amp.autocast():
                raw_outputs, loss = self._forward(data)
                seperate_loss = loss / self.accumulated_iter
            seperate_loss = self.scaler.scale(seperate_loss)
        else:
            raw_outputs, loss = self._forward(data)
            seperate_loss = loss / self.accumulated_iter
        seperate_loss.backward()
        loss = loss.detach().cpu().item()
        if self.verbose:
            self.logger.info(
                "[train {}/{}/{}] loss {}".format(
                    self.current_epoch, index, train_iters, loss
                )
            )

        self.call_hook_func(
            "after_train_iteration_end",
            raw_outputs=raw_outputs,
            loss=loss,
            data=data,
            index=index,
            total_iters=train_iters,
            iteration=self.current_train_iteration,
        )
        return loss

    def _forward(self, data):
        if self.forward_fn:
            return self.forward_fn(self, data)
        else:
            predict = self.model(data[0].to(self.devices))
            loss = self.loss_func(predict, data[1].to(self.devices))
            return predict, loss

    def run_val_iteration(self, index, data, val_iters):
        self.status = "val"
        self.current_val_iteration += 1
        self.call_hook_func(
            "before_val_iteration_start",
            data=data,
            index=index,
            total_iters=val_iters,
            iteration=self.current_val_iteration,
        )
        raw_outputs, loss = self._forward(data)
        loss = loss.detach().cpu().item()
        if self.verbose:
            self.logger.info(
                "[val {}/{}/{}] loss {}".format(
                    self.current_epoch, index, val_iters, loss
                )
            )
        self.call_hook_func(
            "after_val_iteration_ended",
            raw_outputs=raw_outputs,
            loss=loss,
            data=data,
            index=index,
            total_iters=val_iters,
            iteration=self.current_val_iteration,
        )
        return loss

    def persist(self, name):
        """save the model to disk"""
        self.call_hook_func("before_checkpoint_persisted")

        if isinstance(self.model, torch.nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        state = {
            "state_dict": model_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "train_iteration": self.current_train_iteration,
            "val_iteration": self.current_val_iteration,
            "lowest_train_loss": self.lowest_train_loss,
            "lowest_val_loss": self.lowest_val_loss
        }

        for plugin in self.plugins:
            key = f"__plugin.{plugin.__class__.__name__}__"
            state[key] = plugin.state_dict()

        if self.amp and self.amp_scaler:
            state["scaler"] = self.scaler.state_dict()

        modelpath = self.standard_model_path(name)
        torch.save(state, modelpath)
        message = f"save checkpoint to {self.standard_model_path(name)}"
        self.notify(message)
        self.call_hook_func("after_checkpoint_persisted", modelpath=modelpath)

    def standard_model_path(self, model_name):
        return os.path.join(self.models_dir, f"{model_name}.pth.tar")

    def model_file_path(self, model_name):
        model_name_path = Path(str(model_name))
        models_dir_path = Path(self.models_dir)

        search_paths = [
            models_dir_path / f"{model_name}.pth.tar",
            models_dir_path / f"epoch_{model_name}.pth.tar",
            models_dir_path / model_name_path,
            model_name_path,
        ]

        for path in search_paths:
            if path.is_file():
                return path.resolve()

        return None

    # TODO: implement methods below
    def graceful_stop(self):
        """stop train and exist after this epoch"""
        pass

    def save_and_stop(self):
        """save the model immediately and stop training"""
        pass

    def create_dirs(self):
        """Create directories"""
        self.create_dir("")
        self.create_dir(self.code)
        self.create_dir(self.code, "models")

    def create_dir(self, *args):
        """Create directory"""
        current_dir = self.base_dir
        for dir_name in args:
            current_dir = os.path.join(current_dir, dir_name)
            if not os.path.isdir(current_dir):
                os.mkdir(current_dir)
