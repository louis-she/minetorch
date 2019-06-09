import socketio
from minetorch.utils import tail
import eventlet
import glob
from minetorch.orm import Experiment
from pathlib import Path
import urllib.parse

# see https://github.com/miguelgrinberg/python-socketio/issues/155
eventlet.monkey_patch()

sio = socketio.Server(client_manager=socketio.RedisManager('redis://localhost:6379/1'))
app = socketio.WSGIApp(sio)


def parse_query_string(query_string):
    get_arguments = urllib.parse.parse_qs(query_string)
    return {k: v[0] for k, v in get_arguments.items()}


@sio.on('connect', namespace="/common")
def connect(sid, environ):
    experiment_id = parse_query_string(environ.get('QUERY_STRING')).get('experiment_id')
    experiment = Experiment.get(Experiment.id == experiment_id)
    sio.enter_room(sid, experiment.name, namespace="/common")


@sio.on('disconnect', namespace="/common")
def disconnect(sid):
    print('disconnect ', sid)


def create_tail_thread(log_file):
    def tail_thread():
        nonlocal log_file

        def handle_tail(text):
            sio.emit('new_server_log', data=text, namespace="/common", room=str(Path(log_file).parent.stem))
        tail(log_file, handle_tail, 0.4)

    return tail_thread


def get_log_files():
    return set(glob.glob((Path.home() / '.minetorch_server/**/runtime_log.txt').as_posix()))


def watch_thread():
    current_log_files = log_files = get_log_files()
    while True:
        print(log_files)
        for log_file in log_files:
            eventlet.spawn(create_tail_thread(log_file))
        new_log_files = get_log_files()
        log_files = new_log_files - current_log_files
        current_log_files = new_log_files
        eventlet.sleep(10)


eventlet.spawn(watch_thread)
