import torchvision
import torchvision.transforms as transforms


def mean_cifar10(data_loc='./data'):
    """
    Provides insights on the shape, mean and std of dataset
    The mean and std are to be used in the actual preprocessing of the dataset
    https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/8
    :param data_loc: Absolute location of data
    :return: None
    """
    train_transform = transforms.Compose([transforms.ToTensor()])
    data_set = torchvision.datasets.CIFAR10(
        root=data_loc, train=True, download=True,
        transform=train_transform)

    return data_set.data.mean(axis=(0, 1, 2)) / 255


def std_cifar10(data_loc='./data'):
    """
    Provides insights on the shape, mean and std of dataset
    The mean and std are to be used in the actual preprocessing of the dataset
    https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/8
    :param data_loc: Absolute location of data
    :return: None
    """
    train_transform = transforms.Compose([transforms.ToTensor()])
    data_set = torchvision.datasets.CIFAR10(
        root=data_loc, train=True, download=True,
        transform=train_transform)

    return data_set.data.std(axis=(0, 1, 2)) / 255


if __name__ == '__main__':
    print(mean_cifar10(data_loc=r"E:\Datasets"))
    print(std_cifar10(data_loc=r"E:\Datasets"))
