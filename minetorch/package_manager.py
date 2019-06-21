import importlib
from minetorch.utils import server_file
import configparser


class PackageManager():

    def __init__(self):
        self.config_file = server_file('minetorch_config.ini')
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)
        if not self.config.has_section('installed_packages'):
            self.config.add_section('installed_packages')
        if not self.config.has_option('installed_packages', 'packages'):
            packages = ''
        else:
            packages = self.config.get('installed_packages', 'packages')
        packages = list(map(lambda x: x.strip(), packages.split(',')))
        self.packages = list(filter(lambda x: x, packages))

    def list_packages(self):
        print('Current installed packages:')
        for package in self.packages:
            print(package)

    def add_package(self, package_name):
        importlib.import_module(package_name)
        if package_name not in self.packages:
            self.packages.append(package_name)
            self.save_packages()

    def remove_package(self, package_name):
        if package_name in self.packages:
            self.packages.remove(package_name)
            self.save_packages()

    def save_packages(self):
        self.config.set('installed_packages', 'packages', ','.join(self.packages))
        with open(self.config_file, 'w') as f:
            self.config.write(f)


package_manager = PackageManager()

__all__ = ['package_manager']
