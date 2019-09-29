import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from experimental.ACNN_Grouped_ResNet.modded_resnet import BaseResNet
from utilities.data import load_data
from utilities.train_helpers import train, test

if __name__ == '__main__':

    # HyperParams and Others
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 30
    batch_size = 1
    learning_rate = 0.01
    graphs = False
    torch.manual_seed(0)

    # TODO: One Ring to Rule Them All

    # Loading Data
    data_loc = r'E:\Datasets'
    train_loader, test_loader = load_data(data_loc, batch_size, download=False)
    writer = None

    # Tensorboard writer
    if graphs:
        writer = SummaryWriter('/data/summary/mnist_resnet_1')

    # noinspection PyUnresolvedReferences
    model = BaseResNet().to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, 100, writer)
        test(model, device, test_loader)
