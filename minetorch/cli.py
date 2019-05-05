import io
import os
import signal
import subprocess
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

import click
import peewee
import minetorch.core
import minetorch.web as web
from minetorch.orm import Experiment, Component, Model, Dataset, Snapshot
from flask.cli import run_command


def start_web_server():
    os.environ["FLASK_APP"] = web.__file__
    os.environ["FLASK_ENV"] = 'development'
    os.system(f'flask run')


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

    subprocs.append(subprocess.Popen(['flask', 'run'], stdout=sys.stdout))
    subprocs.append(subprocess.Popen(['yarn', 'run', 'dev'], cwd=os.path.dirname(os.path.abspath(web.__file__)), stdout=sys.stdout))

    def signal_handler(sig, frame):
        print('about to kill child processes')
        for proc in subprocs:
            proc.terminate()
        print('all child processes are existed, exist!')
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    for subproc in subprocs:
        subproc.wait()


@cli.command('db:init')
def db_init():
    for model_class in [Experiment, Component, Model, Dataset, Snapshot]:
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


if __name__ == '__main__':
    cli.add_command(run)
    cli.add_command(ls)
    cli()
