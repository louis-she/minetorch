import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='Minetorch',
    description='A tools collection for pytorch users',
    version='0.4.12',
    packages=['minetorch'],
    url="https://github.com/louis-she/minetorch",
    author='louis',
    author_email='chenglu.she@gmail.com',
    keywords='pytorch minecraft',
    install_requires=['tensorboardX', 'torch', 'matplotlib', 'albumentations'],
    entry_points = {
        'console_scripts': ['minetorch=minetorch.command_line:main'],
    }
)

