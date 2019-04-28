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
    print(minetorch.core.registed_models)
    # cli()
