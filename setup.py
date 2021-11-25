import re
import ast
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('minetorch/__init__.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

setuptools.setup(
    name='MineTorch',
    description='A tools collection for pytorch users',
    version=version,
    packages=setuptools.find_packages(),
    include_package_data=True,
    url="https://github.com/minetorch/minetorch",
    author='louis',
    author_email='chenglu.she@gmail.com',
    keywords='pytorch minetorch',
    install_requires=[
        'torch',
        'tqdm',
        'matplotlib',
        'albumentations',
        'ipython',
        'pandas',
        'seaborn'
    ],
    entry_points={
        'console_scripts': ['minetorch=minetorch.command_line:main'],
    }
)
