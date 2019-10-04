import os
from os.path import abspath, dirname

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utilities.train_helpers import get_directories, train
from utilities.cifar_utils.utils import std_cifar10, mean_cifar10
from utilities.visuals import plot_logs, write_csv, write_to_readme

args = {
    'log_dir':'./results',
    'seed':0,
    'dataset': 'CIFAR10',
    'epochs': 1,
    'batch_size': 128,
    'lr': 0.1,
    'graphs':True,
    'csv':True,
    'lr_schedule': 0,
    'momentum': 0.9,
    'nesterov': False,
    'weight_decay': 5e-4,
}


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
    if args['dataset'] == 'CIFAR10':
        
        print('To train and eval on cifar10 dataset......')
        num_classes = 10
        data_loc = './data'

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar10(data_loc), std_cifar10(data_loc)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar10(data_loc), std_cifar10(data_loc)),
        ])
        
        #------------------Load Dataset---------------------------------#

        train_set = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=4)

        test_set = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, num_workers=4)
    else:
        raise NotImplementedError('Only Cifar10 dataset')

    
    torch.manual_seed(args['seed'])

    # logger_dir = os.path.join(dirname(dirname(abspath(__file__))), args_dir['log_dir'])
    logger_dir = args['log_dir']
    if not os.path.exists(logger_dir):
        os.mkdir(logger_dir)
    
    
    #----------------------Initialise model----------------------#

    model = CifarResNet(n=5).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], 
                            momentum=args['momentum'], 
                            nesterov=args['nesterov'], 
                            weight_decay=args['weight_decay'])

    #---------------------Get Logging Directories----------------#

    training_dir, tensorboard_dir, save_models_dir = get_directories(model, args['dataset'], logger_dir)

    #-----------------------Hyperparams-----------------------------#

    device = "cuda:0"
    best_acc = 0
    start_epoch = 0
    criterion = nn.CrossEntropyLoss()

    #--------------------Tensorboard writer---------------------------#

    if args['graphs'] or args['csv']:
        writer = SummaryWriter(tensorboard_dir)  
        train_logger = (np.array([]), np.array([]), np.array([])) 
        test_logger = (np.array([]), np.array([]), np.array([]))
    else:  
        writer, train_logger, test_logger = None, None, None

    #---------------------Start Training-------------------------------#
    
    tick = time.time()
    for epoch in range(start_epoch, args['epochs']):
        train_logger = train(model, device,  # Train Loop
                             train_loader,
                             optimizer, epoch,
                             log_interval=100, writer=writer, logger=train_logger)

        test_logger, acc = test(model, device,  # Evaluation Loop
                                test_loader, epoch,
                                writer=writer, logger=test_logger)
        print('len', len(test_logger))

        #-------------Save best models------------------#
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(save_models_dir, 'best_epoch' + '.pth.tar'))
            torch.save(model.state_dict(), os.path.join(save_models_dir, 'epoch_%s' % epoch + '.pth.tar'))
    run_time = time.time() - tick
    
    #---------------------Save Logs--------------------------------------#

    if args['graphs']:  # Plot log to graphs
        plot_logs(train_logger, training_dir)
        plot_logs(test_logger, training_dir, test=True)

    if args['csv']:  # Save logs to csv
        write_csv(train_logger, training_dir)
        write_csv(test_logger, training_dir, test=True)

    write_to_readme(args['batch_size'], args['lr'], 
                        args['seed'], args['epochs'], run_time, training_dir)