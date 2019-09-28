from agents import ACNN
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import utilities

torch.manual_seed(0)


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

    return train_loader, test_loader, train_dataset, test_dataset


def train(model: nn.Module, device,
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.SGD,
          epoch, log_interval):
    """
    Performs one epoch of training on model
    :param model: Model class
    :param device: Device to train on. Use 'cuda:0' for GPU acceleration
    :param train_loader: Iterable dataset
    :param optimizer: Training agent for performing backprop
    :param epoch: Epoch number for logging purpose
    :param log_interval: Print stats for every
    :return:
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output,_ = model(data)
        loss = F.nll_loss(output, target)  # Negative log likelihood loss
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model: nn.Module, device, 
         test_loader: torch.utils.data.DataLoader):
    """
    Performs evaluation on dataset
    :param model: Model Classs
    :param device: Device to test on. Use 'cuda:0' for GPU acceleration
    :param test_loader:
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':

    # HyperParams and Others
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 3
    batch_size = 64
    learning_rate = 0.01

    # Loading Data
    data_loc = '/home/sachin/Desktop/ACNN/ACNN/data/MNIST'
    train_loader, test_loader, train_dataset, test_dataset = load_data(data_loc, batch_size, download=False)

    model = ACNN(device=device).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval=100)
        test(model, device, test_loader, num_visulaize=1, dir_save_visuals='data/visuals/')

    utilities.visualize(model, device, test_dataset, save_dir='data/visuals', num_visualize=10)