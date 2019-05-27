import contextlib
from pathlib import Path
import os
import eventlet


def tail(file_path, callback):
    current = open(file_path, 'r')
    curino = os.fstat(current.fileno()).st_ino
    while True:
        while True:
            buf = current.read(1024)
            if buf == "":
                break
            callback(buf)
        try:
            if os.stat(file_path).st_ino != curino:
                new = open(file_path, "r")
                current.close()
                current = new
                curino = os.fstat(current.fileno()).st_ino
                continue
        except IOError:
            pass
        eventlet.sleep(1)


def server_file(experiment, file_name):
    if not isinstance(experiment, str):
        experiment = experiment.name

    experiment_dir = Path.home() / '.minetorch_server' / experiment

    try:
        os.makedirs(experiment_dir)
    except FileExistsError:
        pass

    _file = experiment_dir / file_name
    if not os.path.isfile(_file):
        _file.touch()

    return _file.resolve()


def make_runtime_dir(experiment):
    runtime_dir = Path.home() / '.minetorch'
    if not os.path.isdir(runtime_dir):
        os.mkdir(runtime_dir)
    snapshot = experiment.current_snapshot()
    experiment_dir_name = runtime_dir / f"{experiment.name}_{snapshot.id}"
    if not os.path.isdir(experiment_dir_name):
        os.mkdir(experiment_dir_name)
    return experiment_dir_name


@contextlib.contextmanager
def runtime_file(file_name, mode):
    runtime_dir = Path.home() / '.minetorch'
    if not os.path.isdir(runtime_dir):
        os.mkdir(runtime_dir)
    _file = runtime_dir / file_name
    if not os.path.isfile(_file):
        _file.touch()
    f = open(_file, mode)
    yield f
    f.close()
