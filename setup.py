import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()


setuptools.setup(
    name='Minetorch',
    description='A tools collection for pytorch users',
    version='0.3.5',
    packages=setuptools.find_packages(),
    url="https://github.com/louis-she/minetorch",
    author='louis',
    author_email='chenglu.she@gmail.com',
    keywords='pytorch minecraft',
    install_requires=required,
    entry_points = {
        'console_scripts': ['minetorch=minetorch.cli:cli'],
    }
)
