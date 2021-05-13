import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='Minetorch',
    description='A tools collection for pytorch users',
    version='0.6.14',
    packages=setuptools.find_packages(),
    include_package_data=True,
    url="https://github.com/minetorch/minetorch",
    author='louis',
    author_email='chenglu.she@gmail.com',
    keywords='pytorch minetorch',
    install_requires=[
        'tensorboardX',
        'torch',
        'matplotlib',
        'albumentations',
        'seaborn',
        'google-api-python-client',
        'google-auth-httplib2',
        'google-auth-oauthlib'
    ],
    entry_points={
        'console_scripts': ['minetorch=minetorch.command_line:main'],
    }
)
