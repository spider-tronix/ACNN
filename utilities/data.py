import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets


def load_data(data_loc, batch_size, download=True, dataset='MNIST'):
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

    if dataset not in ['MNIST', 'CIFAR10']:
        raise NotImplementedError('Only MNIST/CIFAR10')

    if dataset == 'MNIST':
        t = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((mean_mnist(data_loc=data_loc).item(),),
                                                     (std_mnist(data_loc=data_loc).item(),))])
        train_dataset = datasets.MNIST(root=data_loc, train=True, transform=t, download=download)
        test_dataset = datasets.MNIST(root=data_loc, train=False, transform=t, download=download)

    elif dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar10(data_loc), std_cifar10(data_loc)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar10(data_loc), std_cifar10(data_loc)),
        ])
        train_dataset = datasets.CIFAR10(root=data_loc, train=True, transform=transform_train, download=download)
        test_dataset = datasets.CIFAR10(root=data_loc, train=False, transform=transform_test, download=download)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


def mean_mnist(data_loc='./data'):
    """
    Provides insights on the shape, mean and std of dataset
    The mean and std are to be used in the actual preprocessing of the dataset
    https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/8
    :param data_loc: Absolute location of data
    :return: Mean eg: [0.49139968 0.48215841 0.44653091]
    """
    train_transform = transforms.Compose([transforms.ToTensor()])
    data_set = torchvision.datasets.MNIST(
        root=data_loc, train=True, download=True,
        transform=train_transform)

    return data_set.train_data.float().mean() / 255


def std_mnist(data_loc='./data'):
    """
    Provides insights on the shape, mean and std of dataset
    The mean and std are to be used in the actual preprocessing of the dataset
    https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/8
    :param data_loc: Absolute location of data
    :return: Std eg: [0.24703223 0.24348513 0.26158784]
    """
    train_transform = transforms.Compose([transforms.ToTensor()])
    data_set = torchvision.datasets.MNIST(
        root=data_loc, train=True, download=True,
        transform=train_transform)

    return data_set.train_data.float().std() / 255


def mean_cifar10(data_loc='./data'):
    """
    Provides insights on the shape, mean and std of dataset
    The mean and std are to be used in the actual preprocessing of the dataset
    https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/8
    :param data_loc: Absolute location of data
    :return: Mean eg: [0.49139968 0.48215841 0.44653091]
    """
    train_transform = transforms.Compose([transforms.ToTensor()])
    data_set = torchvision.datasets.CIFAR10(
        root=data_loc, train=True, download=True,
        transform=train_transform)

    return list(data_set.data.mean(axis=(0, 1, 2)) / 255)


def std_cifar10(data_loc='./data'):
    """
    Provides insights on the shape, mean and std of dataset
    The mean and std are to be used in the actual preprocessing of the dataset
    https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/8
    :param data_loc: Absolute location of data
    :return: Std eg: [0.24703223 0.24348513 0.26158784]
    """
    train_transform = transforms.Compose([transforms.ToTensor()])
    data_set = torchvision.datasets.CIFAR10(
        root=data_loc, train=True, download=True,
        transform=train_transform)

    return list(data_set.data.std(axis=(0, 1, 2)) / 255)


def mean_cifar100(data_loc='./data'):
    """
    Provides insights on the shape, mean and std of dataset
    The mean and std are to be used in the actual preprocessing of the dataset
    https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/8
    :param data_loc: Absolute location of data
    :return: Mean eg: [0.50707516 0.48654887 0.44091784]
    """
    train_transform = transforms.Compose([transforms.ToTensor()])
    data_set = torchvision.datasets.CIFAR100(
        root=data_loc, train=True, download=True,
        transform=train_transform)

    return list(np.mean(data_set.data, axis=(0, 1, 2)) / 255)


def std_cifar100(data_loc='./data'):
    """
    Provides insights on the shape, mean and std of dataset
    The mean and std are to be used in the actual preprocessing of the dataset
    https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/8
    :param data_loc: Absolute location of data
    :return: Std eg: [0.26733429 0.25643846 0.27615047]
    """
    train_transform = transforms.Compose([transforms.ToTensor()])
    data_set = torchvision.datasets.CIFAR100(
        root=data_loc, train=True, download=True,
        transform=train_transform)

    return list(np.std(data_set.data, axis=(0, 1, 2)) / 255)
