import socketio
import append_sys_path  # noqa
from minetorch.utils import tail
from pathlib import Path
import eventlet


sio = socketio.Server()
app = socketio.WSGIApp(sio)


@sio.on('connect', namespace="/server_log")
def connect(sid, environ):
    print('client connectted ', sid)


@sio.on('disconnect', namespace="/server_log")
def disconnect(sid):
    print('disconnect ', sid)


def tail_thread():
    def handle_tail(text):
        sio.emit('new_server_log', text, namespace="/server_log")

    test_log_file = Path.home() / '.minetorch_server' / 'new' / 'runtime_log.txt'
    tail(test_log_file, handle_tail)


eventlet.spawn(tail_thread)
