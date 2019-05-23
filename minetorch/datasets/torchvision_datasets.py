from minetorch import dataset, option
from torchvision import datasets, transforms


@dataset('Torchvision MNIST', 'This is a simple wrap for torchvision.datasets.MNIST')
@option('fold', help='Absolute fold path to the dataset', required=True, default="~/.minetorch/torchvision_mnist")
@option('download', type='boolean', default=True)
def mnist(fold, download):
    return datasets.MNIST(fold, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
