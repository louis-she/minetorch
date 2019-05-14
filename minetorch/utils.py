import contextlib
from pathlib import Path
import os

def make_runtime_dir(experiment):
    runtime_dir = Path.home() / '.minetorch'
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
