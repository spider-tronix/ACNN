import torchvision
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

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def _get_mean_std(dataset='MNIST',
                  data_loc=r'E:\Datasets'):
    """
    Provides insights on the shape, mean and std of dataset
    The mean and std are to be used in the actual preprocessing of the dataset
    https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/8
    :param dataset: Name of the dataset
    :param data_loc: Absolute location of data
    :return: None
    """
    if dataset not in ['MNIST', 'SVHN']:
        raise NotImplementedError('Only MNIST/SVHN')

    train_transform = transforms.Compose([transforms.ToTensor()])

    if dataset is 'MNIST':
        data_set = torchvision.datasets.MNIST(
            root=data_loc, train=True, download=True, transform=train_transform)
        print(list(data_set.train_data.size()))
        print(data_set.train_data.float().mean() / 255)
        print(data_set.train_data.float().std() / 255)

    elif dataset is 'SVHN':
        data_set = torchvision.datasets.SVHN(
            root=data_loc, split='train', download=False,
            transform=train_transform)
        print(f"[{data_set.data.size}, {list(next(iter(data_set))[0].shape)}]")
        print(data_set.data.mean() / 255)
        print(data_set.data.std() / 255)
