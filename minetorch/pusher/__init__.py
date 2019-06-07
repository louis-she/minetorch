import socketio
import append_sys_path  # noqa
from minetorch.utils import tail
import eventlet
import re
import glob
from minetorch.orm import Experiment
from pathlib import Path


sio = socketio.Server()
app = socketio.WSGIApp(sio)


@sio.on('connect', namespace="/server_log")
def connect(sid, environ):
    experiment_id = re.findall(r'experiment_id=(.*?)&', environ.get('QUERY_STRING'))[0]
    experiment = Experiment.get(Experiment.id == experiment_id)
    sio.enter_room(sid, experiment.name, namespace="/server_log")


@sio.on('disconnect', namespace="/server_log")
def disconnect(sid):
    print('disconnect ', sid)


def tail_thread(log_file):
    def handle_tail(text):
        sio.emit('new_server_log', text, namespace="/server_log", room=str(Path(log_file).parent.stem))
    tail(log_file, handle_tail, 0.4)


def get_log_files():
    return set(glob.glob((Path.home() / '.minetorch_server/**/runtime_log.txt').as_posix()))


def watch_thread():
    current_log_files = log_files = get_log_files()
    while True:
        for log_file in log_files:
            eventlet.spawn(tail_thread(log_file))
        new_log_files = get_log_files()
        log_files = new_log_files - current_log_files
        current_log_files = new_log_files


eventlet.spawn(watch_thread)
