import logging
import os
import torch
from tensorboardX import SummaryWriter


class Logger(object):

    def __init__(self, log_dir, namespace):
        self.log_dir = os.path.join(log_dir, namespace)
        self.namespace = namespace
        self.models_dir = os.path.join(self.log_dir, 'models')
        self.log_file = os.path.join(self.log_dir, 'log.txt')
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.check_dir('')
        self.check_dir('models')

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.FileHandler(self.log_file))

    def check_dir(self, dirname):
        if not os.path.isdir(os.path.join(self.log_dir, dirname)):
            os.mkdir(os.path.join(self.log_dir, dirname))

    def scalar(self, value, index, data_name):
        key = '{}/{}'.format(self.namespace, data_name)
        if isinstance(value, dict):
            self.writer.add_scalars(key, value, index)
        else:
            self.writer.add_scalar(key, value, index)

    def info(self, log):
        self.logger.info(log)

    def warn(self, log):
        self.logger.warn(log)

    def get_checkpoint(self, name):
        return torch.load(self.model_file_path(name))

    def persist(self, data, name):
        torch.save(data, self.model_file_path(name))
        self.info('save checkpoint to {}'.format(self.model_file_path(name)))

    def model_file_path(self, model_name):
        model_file_name = '{}.pth.tar'.format(model_name)
        return os.path.join(self.models_dir, model_file_name)
