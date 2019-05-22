import logging
from pathlib import Path
from minetorch.runtime import main_process


def main(config_file=None):
    logging_format = '%(levelname)s %(asctime)s %(message)s'
    logging.basicConfig(
        filename=(Path(__file__).parent / 'log.txt').resolve(),
        format=logging_format,
        datefmt="%m-%d %H:%M:%S",
        level=logging.DEBUG
    )

    if not config_file:
        config_file = './config.json'
    main_process(config_file)


if __name__ == '__main__':
    main()
