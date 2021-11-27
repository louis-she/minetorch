import os
import click
import tempfile
from pathlib import Path
from cookiecutter.main import cookiecutter
from subprocess import call


def text_from_editor():
    EDITOR = os.environ.get('EDITOR','vim')
    initial_message = b"""

# Please enter the message for this experiments. This message will
# be saved as `git commit` message so you can check it with `git log`
"""

    with tempfile.NamedTemporaryFile(suffix=".tmp") as tf:
        tf.write(initial_message)
        tf.flush()
        call([EDITOR, tf.name])
        tf.seek(0)
        return tf.read().decode('utf-8')


@click.group()
def cli():
    pass


@cli.group()
def experiment():
    pass


@cli.command()
@click.option("--template", required=True, help='project template', default="mnist")
def new(template="mnist"):
    """Create a MineTorch project"""
    res = cookiecutter((Path(__file__).parent / 'templates' / 'projects' / template).as_posix())
    ret = os.system(f"git init {res}")
    if ret != 0:
        print("git init failed, it's highly recommanded to install git for reproducibility")
        exit(1)
    exit(0)


@cli.command()
@click.option("--template", required=True, default="default", help="plugin template")
def plugin(template="default"):
    """Create a MineTorch plugin"""
    cookiecutter((Path(__file__).parent / 'templates' / 'plugins' / template).as_posix())


@cli.command()
@click.argument("path", required=False, default=".")
def run(path="."):
    """Run a MineTorch project, default run CWD"""
    os.chdir(path)
    ret = os.system("git diff-index --quiet HEAD")
    if ret != 0:
        print("""The project to be runned is not git-clean, please do either of
1. make the repo clean and then `mientorch run ...`
2. using `minetorch submit ...` to run the repo""")
        exit(1)


@cli.command()
@click.argument("path", required=False, default=".")
def run(path="."):
    """Run a MineTorch project, default run CWD"""
    os.chdir(path)
    ret = os.system("git diff-index --quiet HEAD >&2 2>/dev/null")
    if ret != 0:
        print("""The project to be runned is not git-clean, please do either of
1. make the repo clean and then `mientorch run ...`
2. using `minetorch submit ...` to run the repo""")
        exit(1)


@cli.command()
@click.argument("path", required=False, default=".")
@click.option("--message", "-m", required=False)
def submit(path=".", message=""):
    """Submit the changes and run the project"""
    os.chdir(path)

    ret = os.system("git diff-index --quiet HEAD >&2 2>/dev/null")
    if ret != 0:
        os.system("git add -A")
        if message:
            os.system(f"git commit -m {message}")
        else:
            message = "\n".join([line for line in text_from_editor().split("\n") if not line.startswith("#")])
            os.system(f"git commit -m '{message}'")


@experiment.command(name="ls")
def experiment_ls():
    """list experiment"""
    print("list experiments")


if __name__ == "__main__":
    cli()
