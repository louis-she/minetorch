import logging
from minetorch.utils import server_file

runtime_loggers = {}


def get_runtime_logger(experiment):
    global runtime_loggers
    formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s', datefmt="%m-%d %H:%M:%S")
    logger_name = f'runtime_logger_{experiment.id}'

    if logger_name in runtime_loggers:
        return runtime_loggers[logger_name]

    runtime_logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(server_file(experiment, 'runtime_log.txt'))
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    runtime_logger.addHandler(handler)
    runtime_logger.setLevel(logging.DEBUG)
    runtime_loggers[logger_name] = runtime_logger
    return runtime_logger
