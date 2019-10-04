from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from experimental.utils import std_cifar10, mean_cifar10, std_cifar100, mean_cifar100

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
args = {
    'num_gpus': 1,
    'ckpt_dir': 'ckpt/vgg/',
    'dataset': 'cifar100',
    'epochs': 200,
    'batch_size': 200,
    'lr': 0.1,
    'lr_schedule': 0,
    'momentum': 0.9,
    'nesterov': False,
    'weight_decay': 5e-4,
}

if not os.path.exists(args['ckpt_dir']):
    os.makedirs(args['ckpt_dir'])


# noinspection PyShadowingNames
def adjust_learning_rate(optimizer, epoch):
    if args['lr_schedule'] == 0:
        lr = args['lr'] * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 120))
                           * (0.2 ** int(epoch >= 160) * (0.2 ** int(epoch >= 220))))
    elif args['lr_schedule'] == 1:
        lr = args['lr'] * ((0.1 ** int(epoch >= 150))
                           * (0.1 ** int(epoch >= 225)))
    elif args['lr_schedule'] == 2:
        lr = args['lr'] * ((0.1 ** int(epoch >= 80)) *
                           (0.1 ** int(epoch >= 120)))
    else:
        raise Exception("Invalid learning rate schedule!")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# noinspection PyShadowingNames
def train(train_loader, model, criterion, optimizer):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


# noinspection PyShadowingNames,PyShadowingBuiltins
def eval(test_loader, model, epoch, lr, best_acc):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print('Epoch:', epoch, 'Accuracy: %f %%' %
          (100 * correct / total), 'best_accuracy:', best_acc, 'lr:', lr)
    return float(100 * correct / total)


if __name__ == '__main__':
    if args['dataset'] == 'cifar10':
        print('To train and eval on cifar10 dataset......')
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar10, std_cifar10),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar10, std_cifar10),
        ])
        train_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args['batch_size'], shuffle=True,
                                                   num_workers=4)

        test_set = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args['batch_size'], shuffle=False, num_workers=4)
    else:
        print('To train and eval on cifar100 dataset......')
        num_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar100, std_cifar100),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar100, std_cifar100),
        ])
        train_set = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args['batch_size'], shuffle=True,
                                                   num_workers=4)

        test_set = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args['batch_size'], shuffle=False, num_workers=4)

    print('==> Building model..', args['ckpt_dir'][5:])
    model = models.vgg16(num_classes=num_classes)

    model.cuda()
    gpu_ids = range(args['num_gpus'])
    # model = torch.nn.parallel.DataParallel(model, device_ids=gpu_ids)

    # args['snapshot'] = 'best_epoch.pth.tar'
    # print(os.path.join(args['ckpt_dir'], args['snapshot']))
    # model.load_state_dict(torch.load(os.path.join(args['ckpt_dir'], args['snapshot'])))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(
    ), lr=args['lr'], momentum=args['momentum'], nesterov=args['nesterov'], weight_decay=args['weight_decay'])

    best_acc = 0
    start_epoch = 0

    for epoch in range(start_epoch, args['epochs']):
        lr = adjust_learning_rate(optimizer, epoch + 1)

        train(train_loader, model, criterion, optimizer)
        acc = eval(test_loader, model, epoch, lr, best_acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(
                args['ckpt_dir'], 'best_epoch' + '.pth.tar'))
        torch.save(model.state_dict(), os.path.join(
            args['ckpt_dir'], 'epoch_%s' % epoch + '.pth.tar'))
