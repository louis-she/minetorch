import logging
import os
import torch
from tensorboardX import SummaryWriter


class Logger(object):

    def __init__(self, log_dir, namespace):
        self.log_dir = log_dir
        self.namespace = namespace
        self.writer = SummaryWriter()
        self.models_dir = os.path.join(self.log_dir, 'models')
        self.log_file = os.path.join(self.log_dir, 'log.txt')

        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.isdir(self.models_dir):
            os.mkdir(self.models_dir)
        logging.basicConfig(filename=self.log_file, level=logging.DEBUG)

    def scalar(self, value, index, data_name):
        key = '{}/{}'.format(self.namespace, data_name)
        if isinstance(value, str):
            self.writer.add_scalar(key, value, index)
        else:
            self.writer.add_scalars(key, value, index)

    def info(self, log):
        logging.info(log)

    def warn(self, log):
        logging.warn(log)

    def get_checkpoint(self, name):
        torch.load(self.model_file_path(name))

    def persist(self, data, name):
        torch.save(data, self.model_file_path(name))
        self.info('save checkpoint to {}'.format(self.model_file_path(name)))

    def model_file_path(self, model_name):
        model_file_name = '{}.pth.tar'.format(name)
        return os.path.join(self.models_dir, model_file_name)
