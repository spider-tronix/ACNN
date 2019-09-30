import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train(model: nn.Module, device,
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.SGD,
          epoch, log_interval, writer=None):
    """
    Performs one epoch of training on model
    :param model: Model class
    :param device: Device to train on. Use 'cuda:0' for GPU acceleration
    :param train_loader: Iterable dataset
    :param optimizer: Training agent for performing backprop
    :param epoch: Epoch number for logging purpose
    :param log_interval: Print stats for every
    :param writer: Tensorboard writer to track training accuracy
    :return:
    """
    model.train()
    running_loss = 0.0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        data, target = data.to(device), target.to(device)
        
        output = model(data)    # Foward Prop
        loss = F.nll_loss(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    # Backprop
        # Logging
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        running_loss += loss.item()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))

            if writer is not None:
                global_step = (epoch - 1) * len(train_loader) + batch_idx
                writer.add_scalar('Training Loss', running_loss / log_interval, global_step)  # write to tensorboard summary
                writer.add_scalar('Training Accuracy', 100. * correct/(batch_idx * len(data)), global_step)

            running_loss = 0.0

    print('\nTraining Accuracy: {}/{} ({:.4f}%)'.format(correct, len(train_loader.dataset),
                                                        100. * correct / len(train_loader.dataset)))


def test(model: nn.Module, device, test_loader: torch.utils.data.DataLoader, 
        epoch, writer=None):
    """
    Performs evaluation on dataset
    :param model: Model Class
    :param device: Device to test on. Use 'cuda:0' for GPU acceleration
    :param test_loader: Iterable Dataset
    :param epoch: epoch number
    :param writer: Tensorboard writer to track test accuracy
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
    test_acc = 100. * correct / len(test_loader.dataset)

    if writer is not None:  # Logging
        writer.add_scalar('Test Loss (/epoch)', test_loss, epoch)
        writer.add_scalar('Test Accuracy (/epoch)', test_acc, epoch)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))


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
