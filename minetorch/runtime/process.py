import os
import sys
import time
from minetorch.runtime.rpc import RuntimeRpc


def main(settings):
    hey_yo_interval = settings.get('hey_yo_interval', 10)
    while True:
        time.sleep(hey_yo_interval)
        rpc = RuntimeRpc(settings['server_addr'])
        rpc.heyYo()
    sys.exit(0)


def spawn_rpc_process(settings):
    child_pid = os.fork()
    if child_pid == 0:
        main(settings)
    return child_pid
