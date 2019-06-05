import sys
import time
from multiprocessing import Process

import minetorch.constants as C
import minetorch.runtime.process_env as env

current_status = C.STATUS_IDLE
training_process = None


def main_process(config_file):
    global current_status, training_process
    env.init_process_env(config_file)

    hey_yo_interval = env.config.get('hey_yo_interval', 10)
    env.logger.info('runtime main process has started')

    while True:
        res = env.rpc.heyYo(env.config['experiment_id'], current_status)
        if res.command == C.COMMAND_TRAIN and current_status != C.STATUS_TRAINING:
            env.logger.info('start training process')
            current_status = C.STATUS_TRAINING
            training_process = spawn_training_process(config_file)
        elif res.command == C.COMMAND_HALT and current_status != C.STATUS_IDLE:
            env.logger.info('training process has been killed')
            current_status = C.STATUS_IDLE
            training_process.terminate()
        elif res.command == C.COMMAND_KILL:
            env.logger.info('main process has been killed')
            break
        time.sleep(hey_yo_interval)
    sys.exit(0)


def spawn_training_process(config_file):
    from minetorch.runtime.training_process import main as training_process_main
    child_process = Process(target=training_process_main, args=(config_file,))
    child_process.start()
    return child_process
