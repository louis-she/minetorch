import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import tqdm
from IPython.core.display import HTML, display

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
        forward (function, optional):
            custom forward function.
        verbose (boolean, optional):
            los loss of every iteration
    """

    def __init__(
        self,
        alchemistic_directory,
        model,
        optimizer,
        loss_func,
        code="geass",
        train_dataloader=None,
        val_dataloader=None,
        resume=True,
        eval_stride=1,
        persist_stride=1,
        gpu=True,
        drawer="matplotlib",
        hooks={},
        max_epochs=None,
        statable={},
        logging_format=None,
        trival=False,
        in_notebook=False,
        plugins=[],
        logger=None,
        sheet=None,
        accumulated_iter=1,
        ignore_optimizer_resume=False,
        forward=None,
        verbose=False,
        amp=False,
        amp_scaler=True,
    ):
        self.alchemistic_directory = alchemistic_directory
        self.code = code
        if trival:
            self.code = f"trival_{code}"
        self.create_dirs()
        self.gpu = gpu
        self.devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.code_dir = os.path.join(alchemistic_directory, self.code)
        if self.logger is None:
            self.set_logging_config(alchemistic_directory, self.code, logging_format)
            self.logger = logging
        self.create_drawer(drawer)
        self.models_dir = os.path.join(alchemistic_directory, self.code, "models")
        self.in_notebook = in_notebook
        self.statable = statable
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

        self.sheet = sheet
        if self.sheet:
            self._init_sheet()

        self.plugins = plugins
        for plugin in self.plugins:
            plugin.set_miner(self)

        self._set_tqdm()
        self.call_hook_func("before_init")
        self._check_statable()
        self.init_model()
        if self.sheet:
            self.sheet_progress = dict(
                epoch=0, train_percentage="0%", val_percentage="0%"
            )
            self.last_flushed_at = 0
            self.sheet.onready()
            self.sheet.flush()
        self.status = "init"
        self.call_hook_func("after_init")

    def _check_statable(self):
        for name, statable in self.statable.items():
            if not (
                hasattr(statable, "state_dict") and hasattr(statable, "load_state_dict")
            ):
                raise Exception(f"The {name} is not a statable object")

    def _set_tqdm(self):
        if self.in_notebook:
            self.tqdm = tqdm.tqdm_notebook
        else:
            self.tqdm = tqdm.tqdm

    def _init_sheet(self):
        self.sheet.set_miner(self)
        self.sheet.reset_index()
        self.sheet.create_column("code", "Code")
        self.sheet.create_column("progress", "Progress")
        self.sheet.create_column("loss", "Loss")
        self.sheet.update("code", self.code)

    def create_sheet_column(self, key, title):
        if self.sheet is None:
            return
        self.sheet.create_column(key, title)

    def update_sheet(self, key, value):
        if self.sheet is None:
            return
        self.sheet.update(key, value)

    def set_logging_config(self, alchemistic_directory, code, logging_format):
        self.log_dir = os.path.join(alchemistic_directory, code)
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

    def create_drawer(self, drawer):
        if drawer == "tensorboard":
            self.drawer = drawers.TensorboardDrawer(self)
        elif drawer == "matplotlib":
            self.drawer = drawers.MatplotlibDrawer(self)
        else:
            self.drawer = drawer

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
                "`gpu` option is true(default), Minetorch will do this for you."
            )

        if self.resume is True:
            # resume from the newest model
            if self.model_file_path("latest") is not None:
                checkpoint_path = self.model_file_path("latest")
            else:
                checkpoint_path = None
                msg = "Could not find checkpoint to resume, " "train from scratch"
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

            # load drawer state
            if (self.drawer is not None) and ("drawer_state" in checkpoint):
                self.drawer.set_state(checkpoint["drawer_state"])

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

            # load other statable state
            if "statable" in checkpoint:
                for name, statable in self.statable.items():
                    if name not in checkpoint["statable"]:
                        continue
                    statable.load_state_dict(checkpoint["statable"][name])
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
            if not plugin.before_hook(name, payload):
                continue
            if hasattr(plugin, name):
                getattr(plugin, name)(**payload)

    def train(self):
        """start to train the model"""
        while True:
            self.current_epoch += 1
            self.call_hook_func("before_epoch_start", epoch=self.current_epoch)
            self.notebook_divide(f"Epoch {self.current_epoch}")
            self.model.train()
            train_iters = len(self.train_dataloader)

            total_train_loss = 0
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
                    self.optimizer.zero_grad(set_to_none=True)
                total_train_loss += train_loss
                current_percentage = math.ceil(index / total * 100)
                if current_percentage != percentage:
                    self._update_progress(train_percentage=f"{percentage}%")
                    percentage = current_percentage
            self.optimizer.zero_grad(set_to_none=True)
            self._update_progress(force=True, train_percentage=f"{current_percentage}%")

            total_train_loss = total_train_loss / train_iters
            self.notify(
                f"training of epoch {self.current_epoch} finished, "
                f"loss is {total_train_loss}"
            )

            total_val_loss = 0
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
                        total_val_loss += val_loss
                        current_percentage = math.ceil(index / total * 100)
                        if current_percentage != percentage:
                            self._update_progress(val_percentage=f"{percentage}%")
                            percentage = current_percentage
                    self._update_progress(
                        force=True, val_percentage=f"{current_percentage}%"
                    )

                total_val_loss = total_val_loss / val_iters
                self.notify(
                    f"validation of epoch {self.current_epoch}"
                    f"finished, loss is {total_val_loss}"
                )
            if self.drawer is not None:
                png_file = self.drawer.scalars(
                    self.current_epoch,
                    {"train": total_train_loss, "val": total_val_loss},
                    "loss",
                )
                if png_file is not None:
                    self.update_sheet(
                        "loss", {"raw": png_file, "processor": "upload_image"}
                    )

            if total_train_loss < self.lowest_train_loss:
                self.lowest_train_loss = total_train_loss

            if total_val_loss < self.lowest_val_loss:
                message = (
                    "current val loss {} is lower than lowest {}, "
                    "persist this model as best one".format(
                        total_val_loss, self.lowest_val_loss
                    )
                )
                self.notify(message, "success")

                self.lowest_val_loss = total_val_loss
                self.persist("best")
            self.persist("latest")

            if not self.current_epoch % self.persist_stride:
                self.persist("epoch_{}".format(self.current_epoch))

            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.call_hook_func("before_quit")
                self.notify("exceed max epochs, quit!")
                break

            if self.sheet:
                self.sheet.flush()
            self.call_hook_func(
                "after_epoch_end",
                train_loss=total_train_loss,
                val_loss=total_val_loss,
                epoch=self.current_epoch,
            )

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
        if self.amp:
            with torch.cuda.amp.autocast():
                _, loss = self._forward(data)
                seperate_loss = loss / self.accumulated_iter
                if self.amp_scaler:
                    seperate_loss = self.scaler.scale(seperate_loss)
        else:
            _, loss = self._forward(data)
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
        predict, loss = self._forward(data)
        loss = loss.detach().cpu().item()
        if self.verbose:
            self.logger.info(
                "[val {}/{}/{}] loss {}".format(
                    self.current_epoch, index, val_iters, loss
                )
            )
        self.call_hook_func(
            "after_val_iteration_ended",
            predicts=predict,
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
        if self.drawer is not None:
            drawer_state = self.drawer.get_state()
        else:
            drawer_state = {}

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
            "lowest_val_loss": self.lowest_val_loss,
            "drawer_state": drawer_state,
            "statable": {},
        }

        for statable_name, statable in self.statable.items():
            state["statable"][statable_name] = statable.state_dict()

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
            model_name_path,
            models_dir_path / model_name_path,
            models_dir_path / f"{model_name}.pth.tar",
            models_dir_path / f"epoch_{model_name}.pth.tar",
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
        current_dir = self.alchemistic_directory
        for dir_name in args:
            current_dir = os.path.join(current_dir, dir_name)
            if not os.path.isdir(current_dir):
                os.mkdir(current_dir)

    def periodly_flush(self, force=False):
        if self.sheet is None:
            return
        now = int(datetime.now().timestamp())
        # flush every 10 seconds
        if not force and now - self.last_flushed_at < 10:
            return
        self.sheet.flush()
        self.last_flushed_at = now

    def _update_progress(self, force=False, **kwargs):
        if self.sheet is None:
            return

        self.sheet_progress.update(kwargs)
        progress = f"""
         epoch:  {self.sheet_progress.get('epoch')}
train progress:  {self.sheet_progress.get('train_percentage')}
  val progress:  {self.sheet_progress.get('val_percentage')}
"""
        self.sheet.update("progress", progress)
        self.periodly_flush(force)
