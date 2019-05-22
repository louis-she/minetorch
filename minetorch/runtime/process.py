import json
import logging
import sys
import time
from multiprocessing import Process

from minetorch.constants import proto as C
from minetorch.rpc.grpc import minetorch_pb2
from minetorch.runtime.rpc import RuntimeRpc

current_status = minetorch_pb2.HeyMessage.Status.Value('IDLE')
training_process = None


def main_process(config_file):
    global current_status, training_process
    config_file = config_file if config_file else './config.json'
    with open(config_file, 'r') as f:
        config = json.loads(f.read())

    hey_yo_interval = config.get('hey_yo_interval', 10)
    rpc = RuntimeRpc(config['server_addr'])

    while True:
        time.sleep(hey_yo_interval)
        logging.debug('say hey to the server')
        res = rpc.heyYo(config['experiment_id'], current_status)
        logging.debug(f'server respond with: {res}')
        logging.debug(f'server command: {res.command}')

        if res.command == C.COMMAND_TRAIN and current_status != C.STATUS_TRAINING:
            logging.info('start training process!')
            current_status = C.STATUS_TRAINING
            training_process = spawn_training_process(config_file)
        elif res.command == C.COMMAND_HALT and current_status != C.STATUS_IDLE:
            logging.info('kill training process!')
            current_status = C.STATUS_IDLE
            training_process.terminate()
        elif res.command == C.COMMAND_KILL:
            break
    sys.exit(0)


def spawn_training_process(config_file):
    from minetorch.runtime.training_process import main as training_process_main
    child_process = Process(target=training_process_main, args=(config_file,))
    child_process.start()
    return child_process
