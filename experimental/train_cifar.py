import os
import sys
import time
from os import path
from os.path import dirname, abspath

import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(dirname(dirname(abspath(__file__))))

from experimental.utils import std_cifar10, mean_cifar10, std_cifar100, mean_cifar100
from experimental.cleaner_resnet import ACNN
from utilities.train_helpers import get_directories, train, test

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
args = {
    'n1': 9,  # Features Net
    'n2': 5,  # Filters Net,
    'no_classes': 100,  # 100 for CIFAR100, 10 for CIFAR10
    'graphs': True,
    'csv': True,
    'num_gpus': 1,
    'ckpt_dir': 'ckpt/acnn/',
    'dataset': 'cifar100',
    'epochs': 200,
    'batch_size': 256,
    'lr': 0.1,
    'lr_schedule': 0,
    'momentum': 0.9,
    'nesterov': False,
    'weight_decay': 5e-4,
}

if not os.path.exists(args['ckpt_dir']):
    os.makedirs(args['ckpt_dir'])
logger_dir = os.path.join(dirname(dirname(abspath(__file__))), 'results')
if not path.exists(logger_dir):
    os.mkdir(logger_dir)


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


if __name__ == '__main__':
    if args['dataset'] == 'cifar10':
        print('To train and eval on cifar10 dataset......')
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar10(data_loc=r'E:\Datasets'), std_cifar10(data_loc=r'E:\Datasets')),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar10(data_loc=r'E:\Datasets'), std_cifar10(data_loc=r'E:\Datasets')),
        ])

        train_set = torchvision.datasets.CIFAR10(
            root=r'E:\Datasets', train=True, download=True, transform=transform_train)
        train_loader = DataLoader(
            train_set, batch_size=args['batch_size'], shuffle=True,
            # num_workers=4
        )
        test_set = torchvision.datasets.CIFAR10(
            root=r'E:\Datasets', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(
            test_set, batch_size=args['batch_size'], shuffle=False,
            # num_workers=4
        )

    else:
        print('To train and eval on cifar100 dataset......')
        num_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar100(data_loc=r'E:\Datasets'), std_cifar100(data_loc=r'E:\Datasets')),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar100(data_loc=r'E:\Datasets'), std_cifar100(data_loc=r'E:\Datasets')),
        ])

        train_set = torchvision.datasets.CIFAR100(
            root=r'E:\Datasets', train=True, download=True, transform=transform_train)
        train_loader = DataLoader(
            train_set, batch_size=args['batch_size'], shuffle=True,
            # num_workers=4
        )
        test_set = torchvision.datasets.CIFAR100(
            root=r'E:\Datasets', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args['batch_size'], shuffle=False,
            # num_workers=4
        )

    print('==> Building model..', args['ckpt_dir'][5:])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ACNN(
        n1=args['n1'],
        n2=args['n2'],
        no_classes=args['no_classes']
    )
    model.to(device=device)

    # gpu_ids = range(args['num_gpus'])
    # model = torch.nn.parallel.DataParallel(model, device_ids=gpu_ids)
    # args['snapshot'] = 'best_epoch.pth.tar'
    # print(os.path.join(args['ckpt_dir'], args['snapshot']))
    # model.load_state_dict(torch.load(os.path.join(args['ckpt_dir'], args['snapshot'])))

    optimizer = optim.SGD(
        model.parameters(),
        lr=args['lr'], momentum=args['momentum'],
        weight_decay=args['weight_decay'],
        nesterov=args['nesterov']
    )

    best_acc = 0
    start_epoch = 0

    training_dir, tensorboard_dir = get_directories(model, args['dataset'].upper(), logger_dir)
    # Tensorboard writer
    if args['graphs'] or args['csv']:
        writer = SummaryWriter(tensorboard_dir)  # Handles the creation of tensorboard logs
        train_logger = (np.array([]), np.array([]), np.array([]))  # Handles the creation of png, csv logs
        test_logger = (np.array([]), np.array([]), np.array([]))
    else:  # Avoid weak warnings
        writer, train_logger, test_logger = None, None, None

    tick = time.time()
    for epoch in range(1, args['epochs'] + 1):
        lr = adjust_learning_rate(optimizer, epoch + 1)

        train_logger = train(model, device,  # Train Loop
                             train_loader,
                             optimizer, epoch,
                             log_interval=100, writer=writer, logger=train_logger)

        test_logger = test(model, device,  # Evaluation Loop
                           test_loader, epoch,
                           writer=writer, logger=test_logger)
    run_time = time.time() - tick

    # TODO: The part below to be integrated into train loop
    # if acc > best_acc:
    #     best_acc = acc
    #     torch.save(model.state_dict(), os.path.join(
    #         args['ckpt_dir'], 'best_epoch' + '.pth.tar'))
    # torch.save(model.state_dict(), os.path.join(
    #     args['ckpt_dir'], 'epoch_%s' % epoch + '.pth.tar'))
