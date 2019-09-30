import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets


def load_data(data_loc, batch_size, download=False, dataset='MNIST'):
    """
    Downloads data or uses existing. Packs into iterable Dataloader
    :param dataset: Name of Dataset (MNIST | SVHN)
    :param data_loc: Location to search for existing data or download if absent
    :param batch_size: Number os examples in a single batch
    :param download: Set flag to download data
    :return: Train Loader and Test Loader of dtype torch.utilities.data.dataloader
    """
    train_dataset = None
    test_dataset = None

    if dataset not in ['MNIST', 'SVHN', 'CIFAR10']:
        raise NotImplementedError('Only MNIST/SVHN/CIFAR10')

    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if dataset == 'MNIST':
        train_dataset = datasets.MNIST(root=data_loc, train=True, transform=t, download=download)
        test_dataset = datasets.MNIST(root=data_loc, train=False, transform=t, download=download)
    elif dataset == 'SVHN':
        train_dataset = datasets.SVHN(root=data_loc, split='train', transform=t, download=download)
        test_dataset = datasets.SVHN(root=data_loc, split='test', transform=t, download=download)
    elif dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root=data_loc, train=True, transform=t, download=download)
        test_dataset = datasets.CIFAR10(root=data_loc, train=False, transform=t, download=download)

    return train_loader, test_loader
