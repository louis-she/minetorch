import sys
sys.path.append('..')

import os
import minetorch.web as web
import click
import minetorch.core
from flask.cli import run_command


def start_web_server():
    os.environ["FLASK_APP"] = web.__file__
    sys.argv = sys.argv[0:1]
    run_command()


@click.group()
def cli():
    pass

@cli.command('ls')
def ls():
    minetorch.core.boot()

@cli.command('run')
def run():
    minetorch.core.boot()
    start_web_server()

cli.add_command(run)
cli.add_command(ls)

if __name__ == '__main__':
    # model = minetorch.model.registed_models[0]
    # parameters = {'size': '34'}
    # model.model_class(**parameters)
    # os.environ["FLASK_APP"]='web/web.py'
    cli()
