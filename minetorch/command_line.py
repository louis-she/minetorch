import sys
sys.path.append('..')

import click
import minetorch.core

@click.group()
def cli():
    pass

@click.command()
def ls():
    minetorch.core.boot()

cli.add_command(ls)

if __name__ == '__main__':
    minetorch.core.boot()
    model = minetorch.ModelDecorator.registed_models[0]
    parameters = {'size': '34'}
    model.model_class(**parameters)
    # cli()
