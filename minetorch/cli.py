import os
import signal
import subprocess
import sys
from pathlib import Path

import append_sys_path  # noqa: F401
import click
import minetorch.core
import minetorch.rpc_server
import minetorch.web as web
from minetorch.orm import Component, Experiment, Snapshot


PYTHON_INTERPRETER = Path(__file__).parent.resolve() / '.venv/bin/python'


def start_web_server():
    os.environ["FLASK_APP"] = web.__file__
    os.environ["FLASK_ENV"] = 'development'
    os.system(f'flask run')


def start_rpc_server():
    from rpc import RpcServer
    child_pid = os.fork()
    if child_pid == 0:
        server = RpcServer(10, '[::]:50051')
        server.serve()
    if child_pid != 0:
        return child_pid


def start_webpack():
    os.system(f'cd {os.path.dirname(web.__file__)}; npx webpack')


@click.group()
def cli():
    pass


@cli.command('dev')
def development():
    subprocs = []

    os.environ["FLASK_APP"] = web.__file__
    os.environ["FLASK_ENV"] = 'development'
    start_rpc_server()

    subprocs.append(subprocess.Popen(
        [PYTHON_INTERPRETER, '-m', 'flask', 'run'],
        stdout=sys.stdout,
        cwd=Path(__file__).parent
    ))
    subprocs.append(subprocess.Popen(
        ['yarn', 'run', 'dev'],
        cwd=Path(web.__file__).parent,
        stdout=sys.stdout
    ))

    def signal_handler(sig, frame):
        print('about to kill child processes')
        for proc in subprocs:
            proc.terminate()
        print('all child processes are existed, exist!')
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    os.wait()


@cli.command('db:init')
def db_init():
    for model_class in [Experiment, Component, Snapshot]:
        model_class.drop_table()
        print(f"creating {model_class}")
        model_class.create_table(safe=False)
        print(f"{model_class} created")


@cli.command('ls')
def ls():
    minetorch.core.boot()


@cli.command('run')
def run():
    minetorch.core.boot()
    start_web_server()


@cli.command('proto:compile')
def proto_compile():
    minetorch_dir = Path(__file__).resolve().parents[1]
    proto_dir = Path('minetorch') / 'rpc' / 'grpc'
    subprocess.Popen([
        PYTHON_INTERPRETER,
        "-m",
        "grpc_tools.protoc",
        f"-I.",
        f"--python_out=.",
        f"--grpc_python_out=.",
        f"{proto_dir / 'minetorch.proto'}"
    ], stdout=sys.stdout, cwd=minetorch_dir)


if __name__ == '__main__':
    # cli.add_command(run)
    # cli.add_command(ls)
    cli()
