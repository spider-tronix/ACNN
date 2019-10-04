import torchvision
import torchvision.transforms as transforms
import numpy as np


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


if __name__ == '__main__':
    print(mean_cifar10(data_loc=r"E:\Datasets"))
    print(std_cifar10(data_loc=r"E:\Datasets"))
    print(mean_cifar100(data_loc=r"E:\Datasets"))
    print(std_cifar100(data_loc=r"E:\Datasets"))
