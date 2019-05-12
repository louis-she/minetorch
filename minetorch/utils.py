import contextlib
from pathlib import Path
import os

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
