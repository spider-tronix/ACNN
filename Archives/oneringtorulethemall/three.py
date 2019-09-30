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
    :param logger: Tuple of numpy array to log values 
    :return: new logger
    """
    if logger is not None:
        step_log, loss_log, acc_log = logger
    else:
        step_log, loss_log, acc_log = None, None, None

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
                step_log = np.append(step_log, global_step)
                acc_log = np.append(acc_log, a)
                loss_log = np.append(loss_log, l)

            running_loss = 0.0

    print('\nTraining Accuracy: {}/{} ({:.4f}%)'.format(correct, len(train_loader.dataset),
                                                        100. * correct / len(train_loader.dataset)))

    return step_log, loss_log, acc_log


def test(model: nn.Module, device, test_loader: torch.utils.data.DataLoader,
         epoch, writer=None, logger=None):
    """
    Performs evaluation on dataset
    :param model: Model Class
    :param device: Device to test on. Use 'cuda:0' for GPU acceleration
    :param test_loader: Iterable Dataset
    :param epoch: epoch number
    :param writer: Tensorboard writer to track test accuracy
    :param logger: Tuple of numpy array to log values
    :return:
    """

    if logger is not None:
        step_log, loss_log, acc_log = logger
    else:
        step_log, loss_log, acc_log = None, None, None

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
    test_acc = 100. * correct / len(test_loader.dataset)

    if writer is not None:  # Tensorboard Logging
        writer.add_scalar('Test Loss (/epoch)', test_loss, epoch)
        writer.add_scalar('Test Accuracy (/epoch)', test_acc, epoch)

    if logger is not None:  # Manual Logging
        step_log = np.append(step_log, epoch)
        acc_log = np.append(acc_log, test_acc)
        loss_log = np.append(loss_log, test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))

    return step_log, loss_log, acc_log
