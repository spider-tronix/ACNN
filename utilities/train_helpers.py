import os
from os import path

import numpy as np
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train(model: nn.Module, device,
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.SGD,
          criterion,
          epoch, log_interval, writer=None, logger=None):
    """
    Performs one epoch of training on model
    :param criterion: Loss Function
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

        output = model(data)  # Forward Prop
        loss = criterion(output, target)

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

    return (step_log, loss_log, acc_log)


def test(model: nn.Module, device, test_loader: torch.utils.data.DataLoader,
         epoch, criterion, writer=None, logger=None, ):
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
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
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

    return (step_log, loss_log, acc_log), test_acc


# noinspection PyShadowingNames
def grouped_conv(img, filters):
    batch_size, c_in, h1, w1 = img.shape
    _, c_out, h2, w2 = filters.shape

    filters = filters[:, :, None, :, :]
    filters = filters.repeat(1, 1, c_in, 1, 1)

    return F.conv2d(
        input=img.view(1, batch_size * c_in, h1, w1),
        weight=filters.view(batch_size * c_out, c_in, h2, w2),
        groups=batch_size
    ).reshape(batch_size, -1)


def default_config(model, dataset):
    """
    Returns default layers used for testing that model works.
    :param model: Model Class
    :param dataset: Dataset Used for Training model
    :return: default channels, kernel sizes and strides for both networks
    """
    if model not in ['Vanilla_ACNN', 'ACNN_ResNet']:
        raise NotImplementedError('Only Vanilla_ACNN/ACNN_ResNet')

    if dataset not in ['MNIST', 'CIFAR10', 'SVHN']:
        raise NotImplementedError('Only MNIST/SVHN/CIFAR10')

    if model == 'Vanilla_ACNN':
        if dataset == 'MNIST':
            net1_channels = (1, 16, 32)
            net2_channels = (1, 16, 32, 64)
            net1_kernels_size = (3, 5)
            net2_kernels_size = (3, 5, 5)
            net1_strides = (1, 2)
            net2_strides = (1, 2, 2)
            fc_units = (4096, 1024, 256, 64, 10)
        elif dataset == 'CIFAR10':
            pass
        elif dataset == 'SVHN':
            pass
    elif model == 'ACNN_ResNet':
        if dataset == 'MNIST':
            pass
        elif dataset == 'CIFAR10':
            pass
        elif dataset == 'SVHN':
            pass

    return net1_channels, net2_channels, \
           net1_kernels_size, net2_kernels_size, \
           net1_strides, net2_strides, \
           fc_units


def get_directories(model, dataset, parent_dir):
    """
    Returns paths to save training logs
    :param model: Instance of the model class
    :param dataset: String containing dataset name
    :param parent_dir: dir where sub directories are to be made
    :return: Training_dir and Tensorboard_dir, Save_models_dir for saving required files
    """

    if dataset not in ['MNIST', 'CIFAR10', 'SVHN', 'CIFAR100']:
        raise NotImplementedError('Only MNIST/SVHN/CIFAR10/CIFAR100')

    model_name = model.__class__.__name__
    parent_dir = os.path.abspath(parent_dir)
    dataset_dir = os.path.join(parent_dir, dataset)
    model_dir = os.path.join(dataset_dir, model_name)

    if not path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    if not path.exists(model_dir):
        os.mkdir(model_dir)

    i = 1
    while True:
        training_dir = os.path.join(model_dir, f'Training_{i}')
        if not path.exists(training_dir):  # TODO: next(os.walk(model_dir))[1] is more optimized. To be replaced with it
            os.mkdir(training_dir)
            break
        i += 1

    tensorboard_dir = os.path.join(training_dir, 'Tensorboard_Summary')
    save_models_dir = os.path.join(training_dir, 'Saved_Models')

    if not path.exists(save_models_dir):
        os.mkdir(save_models_dir)

    return training_dir, tensorboard_dir, save_models_dir
