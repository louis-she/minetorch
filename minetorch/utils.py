import contextlib
from pathlib import Path
import os


@contextlib.contextmanager
def minetorch_server_file(experiment, file_name, mode='a'):
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
    f = open()



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
