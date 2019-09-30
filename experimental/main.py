import os
import time
from os import path
import pandas as pd 

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from experimental.one import get_directories
from experimental.two import write_to_Readme
from experimental.three import train

from models.Vanilla_Acnn import VanillaACNN
from utilities.data import load_data
from utilities.train_helpers import test, default_config
from utilities.visuals import plot_logs

if __name__ == '__main__':
    dataset = 'MNIST'

    # HyperParams and Others
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 0
    epochs = 3
    batch_size = 64
    learning_rate = 0.01

    graphs = True
    logger_dir = 'testing2/'

    download = False
    torch.manual_seed(seed)

    # Loading Data
    data_loc = r'E:\Datasets'
    train_loader, test_loader = load_data(data_loc, batch_size, download=download, dataset=dataset)

    # noinspection PyUnresolvedReferences
    model = VanillaACNN(*default_config(dataset=dataset, model=model)).to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    if not path.exists(logger_dir):
        os.mkdir(logger_dir)
    training_dir, tensorboard_dir = get_directories(model, dataset, logger_dir)

    # Tensorboard writer
    if graphs:
        writer = SummaryWriter(tensorboard_dir)  # Handles the creation of tensorboard logs
        train_logger = (np.array([]), np.array([]), np.array([]))  # Handles the creation of png, csv logs
    else:  # Avoid weak warnings
        writer = None
        train_logger = None

    tick = time.time()
    for epoch in range(1, epochs + 1):
        train_logger = train(model, device,  # Train Loop
                             train_loader,
                             optimizer, epoch,
                             log_interval=100, writer=writer, logger=train_logger)
        test(model, device,  # Evaluation Loop
             test_loader, epoch,
             writer=writer)
    run_time = tick - time.time()   

    if train_logger is not None:    # Save log plots
        plot_logs(train_logger, training_dir)
    
    write_to_Readme(batch_size, learning_rate, seed, epochs, training_dir) # write to Readme.md