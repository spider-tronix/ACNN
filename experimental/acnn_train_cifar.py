import argparse
import os
import sys
import time
from os.path import abspath, dirname

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

sys.path.append(dirname(dirname(abspath(__file__))))

from utilities.train_helpers import get_directories, test, train
from utilities.visuals import plot_logs, write_csv, write_to_readme
from experimental.acnn_resnet_cifar import ACNN
from benchmarks.models.cifar_resnet_v2 import BenchmarkResNet
from utilities.data import load_data


def parse_train_args():
    parser = argparse.ArgumentParser("Features Network --> ResNet, Filters network --> Drastic Convolution")
    parser.add_argument("--n1", default=18, help="ResNet depth for Features Network")
    parser.add_argument("--n2", default=3, help="ResNet depth for Filters Network")
    parser.add_argument("--bs", default=128, help="Batch Size for training data")
    parser.add_argument("--dataset", default='CIFAR10', help="Dataset for Training")
    parser.add_argument("--epochs", default=1, help="Number of epochs")
    parser.add_argument("--lr", default=0.1, help="Initial Learning Rate")
    parser.add_argument("--logs", default=True, help="TensorBoard Logging")
    parser.add_argument("--log-dir", default='./results', help="Directory to save logs")
    parser.add_argument("--seed", default=0, help="value of torch.manual_seed")
    parser.add_argument("--lr-schedule", default=0, help="LR Scheduler profile to adjust learning rate")
    parser.add_argument("--momentum", default=0.9, help="Momentum for SGD")
    parser.add_argument("--nesterov", default=False, help="Perform Nesterov gamble correction approach to learning")
    parser.add_argument("--weight-decay", default=5e-4, help="Weight decay for SGD")
    return parser.parse_args()


# noinspection PyShadowingNames
def adjust_learning_rate(args, optimizer, epoch):
    # TODO: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    if epoch <= 3:  # Accelerating initial mini batches as warm up
        lr = 0.01
    elif args.lr_schedule == 0:
        lr = args.lr * ((0.2 ** int(epoch >= 60)) *
                        (0.2 ** int(epoch >= 120)) *
                        (0.2 ** int(epoch >= 160)) *
                        (0.2 ** int(epoch >= 220)))
    elif args.lr_schedule == 1:
        lr = args.lr * ((0.1 ** int(epoch >= 150)) *
                        (0.1 ** int(epoch >= 225)))
    elif args.lr_schedule == 2:
        lr = args.lr * ((0.1 ** int(epoch >= 80)) *
                        (0.1 ** int(epoch >= 120)))
    else:
        raise Exception("Invalid learning rate schedule!")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    args = parse_train_args()

    torch.manual_seed(args.seed)

    # ----------------------Load Data---------------------- #
    if args.dataset == 'CIFAR10':
        print('To train and eval on cifar10 dataset......')
        num_classes = 10
        data_loc = r'./data'

        train_loader, test_loader = load_data(data_loc, args.bs, dataset=args.dataset, download=False)
    else:
        raise NotImplementedError('Only Cifar10 dataset')

    # ----------------------Initialise model---------------------- #
    # noinspection PyUnresolvedReferences
    # model = ACNN(n1=args.n1,
    #              n2=args.n2).cuda()

    model = BenchmarkResNet(18).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          nesterov=args.nesterov,
                          weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # ----------------------Get Logging Directories---------------------- #
    # logger_dir = os.path.join(dirname(dirname(abspath(__file__))), args_dir['log_dir'])
    logger_dir = args.log_dir
    if not os.path.exists(logger_dir):
        os.mkdir(logger_dir)
    training_dir, tensorboard_dir, save_models_dir = get_directories(model, args.dataset, logger_dir)

    # ----------------------Tensorboard writer---------------------- #
    if args.logs:
        writer = SummaryWriter(tensorboard_dir)
        train_logger = (np.array([]), np.array([]), np.array([]))
        test_logger = (np.array([]), np.array([]), np.array([]))
    else:
        writer, train_logger, test_logger = None, None, None

    # ----------------------Start Training---------------------- #
    device = "cuda:0"
    best_acc = 0
    start_epoch = 1

    tick = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        # noinspection PyTypeChecker
        lr = adjust_learning_rate(args, optimizer, epoch)
        print(f"Learning Rate: {lr}")
        train_logger = train(model, device,  # Train Loop
                             train_loader,
                             optimizer, epoch=epoch, criterion=criterion,
                             log_interval=100, writer=writer, logger=train_logger)

        test_logger, acc = test(model, device,  # Evaluation Loop
                                test_loader, epoch,
                                criterion=criterion,
                                writer=writer, logger=test_logger)

        # ----------------------Save best models---------------------- #
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(save_models_dir, 'best_epoch' + '.pth.tar'))
            torch.save(model.state_dict(), os.path.join(save_models_dir, 'epoch_%s' % epoch + '.pth.tar'))
    run_time = time.time() - tick

    # ----------------------Save Logs---------------------- #

    if args.logs:  # Plot log to graphs
        plot_logs(train_logger, training_dir)
        plot_logs(test_logger, training_dir, test=True)
        write_csv(train_logger, training_dir)
        write_csv(test_logger, training_dir, test=True)

    write_to_readme(args.batch_size, args.lr,
                    args.seed, args.epochs, run_time, training_dir)
