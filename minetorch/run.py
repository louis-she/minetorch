from minetorch.runtime import main_process
from dotenv import load_dotenv

load_dotenv()


def main(config_file=None):
    if not config_file:
        config_file = './config.json'
    main_process(config_file)


if __name__ == '__main__':
    main()
