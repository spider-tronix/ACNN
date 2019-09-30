import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


def train(model: nn.Module, device,
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.SGD,
          epoch, log_interval, writer=None, logger=None):
    """
    Performs one epoch of training on model
    :param logger:
    :param model: Model class
    :param device: Device to train on. Use 'cuda:0' for GPU acceleration
    :param train_loader: Iterable dataset
    :param optimizer: Training agent for performing backprop
    :param epoch: Epoch number for logging purpose
    :param log_interval: Print stats for every
    :param writer: Tensorboard writer to track training accuracy
    :return: new logger
    """
    if logger is not None:
        step, train_loss, train_accuracy = logger
    else:
        step, train_loss, train_accuracy = None, None, None

    model.train()
    running_loss = 0.0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        data, target = data.to(device), target.to(device)

        output = model(data)  # Foward Prop
        loss = F.nll_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # Backprop
        # Logging
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        running_loss += loss.item()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))

            global_step = (epoch - 1) * len(train_loader) + batch_idx
            l = running_loss / log_interval
            a = 100. * correct / (batch_idx * len(data))

            if writer is not None:
                writer.add_scalar('Training Loss', l,
                                  global_step)  # write to tensorboard summary
                writer.add_scalar('Training Accuracy', a, global_step)

            if logger is not None:
                np.append(step, global_step)
                np.append(train_accuracy, a)
                np.append(train_loss, l)

            running_loss = 0.0

    print('\nTraining Accuracy: {}/{} ({:.4f}%)'.format(correct, len(train_loader.dataset),
                                                        100. * correct / len(train_loader.dataset)))

    return step, train_loss, train_accuracy
