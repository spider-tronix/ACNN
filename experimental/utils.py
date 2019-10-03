import torchvision
import torchvision.transforms as transforms
import numpy as np


def _get_mean_std(dataset='cifar10',
                  data_loc=r'E:\Datasets'):
    """
    Provides insights on the shape, mean and std of dataset
    The mean and std are to be used in the actual preprocessing of the dataset
    https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/8
    :param dataset: Name of the dataset
    :param data_loc: Absolute location of data
    :return: None
    """
    if dataset not in ['cifar10', 'cifar100']:
        raise NotImplementedError('Only CIFAR10 | 100')

    mean, std = None, None
    train_transform = transforms.Compose([transforms.ToTensor()])

    if dataset is 'cifar10':
        data_set = torchvision.datasets.CIFAR10(
            root=data_loc, train=True, download=True,
            transform=train_transform)
        mean = (data_set.data.mean(axis=(0, 1, 2)) / 255)
        std = (data_set.data.std(axis=(0, 1, 2)) / 255)
    elif dataset is 'cifar100':
        data_set = torchvision.datasets.CIFAR100(
            root=data_loc, train=True, download=True,
            transform=train_transform)
        mean = (np.mean(data_set.data, axis=(0, 1, 2)) / 255)
        std = (np.std(data_set.data, axis=(0, 1, 2)) / 255)
    return mean, std


print(_get_mean_std('cifar10'))
