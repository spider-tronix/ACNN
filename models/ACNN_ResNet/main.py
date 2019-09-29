import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from agents import AcnnResNet
from utilities.data import load_data
from utilities.train_helpers import train, test

if __name__ == '__main__':

    # HyperParams and Others
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 3
    batch_size = 64
    learning_rate = 0.01

    # Loading Data
    data_loc = r'E:\Datasets'
    train_loader, test_loader = load_data(data_loc, batch_size, download=False)

    # Tensorboard writer
    writer = SummaryWriter('/data/summary/mnist_resnet_1')

    # noinspection PyUnresolvedReferences
    model = AcnnResNet(device=device).to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, 1, writer)
        test(model, device, test_loader)
