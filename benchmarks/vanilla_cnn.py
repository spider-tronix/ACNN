import torch
from torch import optim

from benchmarks.agents.vanilla_CNN.base_model import BaseModel
from utilities.data import load_data
from utilities.train_helpers import train, test

if __name__ == '__main__':

    batch_size = 64
    lr = 0.01
    epochs = 3

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_loc = r'E:\Datasets'

    train_loader, test_loader = load_data(data_loc, batch_size)

    # noinspection PyUnresolvedReferences
    model = BaseModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval=100)
        test(model, device, test_loader)
