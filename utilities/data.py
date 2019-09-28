import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets


def load_data(data_loc, batch_size, download=False):
    """
    Downloads data or uses existing. Packs into iterable Dataloader
    :param data_loc: Location to search for existing data or download if absent
    :param batch_size: Number os examples in a single batch
    :param download: Set flag to download data
    :return: Train Loader and Test Loader of dtype torch.utilities.data.dataloader
    """
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root=data_loc, train=True, transform=t, download=download)
    test_dataset = datasets.MNIST(root=data_loc, train=False, transform=t, download=download)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
