import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.ACNN_Grouped_CNN.grouped_conv import ACNN
from utilities.data import load_data
from utilities.train_helpers import train, test

if __name__ == '__main__':
    dataset = 'SVHN'

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

    input_channels = next(iter(train_loader))[0].shape[1]

    # Tensorboard writer
    if graphs:
        writer = SummaryWriter('/data/summary/mnist_resnet_1')

    # noinspection PyUnresolvedReferences
    model = ACNN(input_channels=input_channels).to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, 100, writer)
        test(model, device, test_loader)
