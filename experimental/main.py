import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.Vanilla_Acnn import ACNN
from utilities.data import load_data
from utilities.train_helpers import test, default_config
from experimental.three import train


if __name__ == '__main__':
    dataset = 'MNIST'
    model = 'Vanilla_ACNN'

    # HyperParams and Others
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 30
    batch_size = 64
    learning_rate = 0.01
    graphs = False
    download = False
    torch.manual_seed(0)

    # TODO: One Ring to Rule Them All

    # Loading Data
    data_loc = r'E:\Datasets'
    train_loader, test_loader = load_data(data_loc, batch_size, download=download, dataset=dataset)
    writer = None
    train_logger = None

    # Tensorboard writer
    if graphs:
        writer = SummaryWriter('/data/summary/mnist_resnet_1')

        step = np.array([])
        train_loss = np.array([])
        train_accuracy = np.array([])
        train_logger = (step, train_loss, train_accuracy)

    model = ACNN(*default_config(dataset=dataset, model=model)).to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train_logger = train(model, device, train_loader, optimizer, epoch, 100, writer, train_logger)
        test(model, device, test_loader, epoch, writer)
