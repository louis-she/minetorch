import torch
import os

class Trainer(object):

    def __init__(self, logger, model, optimizer, loss_func,
                 train_dataloader=None, val_dataloader=None,
                 resume=True, eval_stride=1, persist_stride=1,
                 hooks={}, max_epochs=None):
        self.logger = logger
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

    def init_model(self):
        """resume from some checkpoint
        """
        if self.resume is True:
            # resume from the newest model
            if os.path.isfile(self.logger.model_file_path('latest')):
                checkpoint = self.logger.get_checkpoint('latest')
            else:
                checkpoint = None
                self.logger.warn('Could not find checkpoint to resume, train '
                                 'from scratch')
        elif isinstance(self.resume, str):
            checkpoint = self.logger.get_checkpoint(self.resume)
        else:
            checkpoint = False

        if checkpoint:
            self.logger.info("Start to load checkpoint")
            self.current_epoch = checkpoint['epoch']
            self.lowest_train_loss = checkpoint['lowest_train_loss']
            self.lowest_val_loss = checkpoint['lowest_val_loss']

            try:
                self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            except:
                self.logger.warn(
                    'load checkpoint failed, the state in the '
                    'checkpoint is not matched with the model, '
                    'try to reload checkpoint with unstrict mode')
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info('Checkpoint loaded')

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

            self.logger.scalar(
                {'train': train_loss, 'val': val_loss},
                self.current_epoch,
                'loss'
            )

            if train_loss < self.lowest_train_loss:
                self.lowest_train_loss = train_loss

            if val_loss < self.lowest_val_loss:
                self.logger.info(
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
                self.logger.info('exceed max epochs, quit!')
                break

    def run_train_iteration(self, index, data, train_iters):
        loss = self.loss_func(self.model, data, self.logger)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.logger.info('[train {}/{}/{}] loss {}'.format(
            self.current_epoch, index, train_iters, loss))

        if loss < self.lowest_train_loss:
            self.lowest_train_loss = loss

        return loss

    def run_val_iteration(self, index, data, val_iters):
        loss = self.loss_func(self.model, data, self.logger)
        self.logger.info('[val {}/{}/{}] loss {}'.format(
            self.current_epoch, index, val_iters, loss))

        return loss

    def persist(self, name):
        """save the model to disk
        """
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'lowest_train_loss': self.lowest_train_loss,
            'lowest_val_loss': self.lowest_val_loss
        }

        self.logger.persist(state, name)

    # TODO: implement methods below
    def graceful_stop(self):
        """stop train and exist after this epoch
        """
        pass

    def save_and_stop(self):
        """save the model immediately and stop training
        """
        pass
