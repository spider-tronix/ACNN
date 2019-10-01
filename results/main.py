import os
import sys
import time
from os import path
from os.path import dirname, abspath

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(dirname(dirname(abspath(__file__))))
from models.Vanilla_Acnn import VanillaACNN
from utilities.data import load_data
from utilities.train_helpers import default_config, get_directories, train, test
from utilities.visuals import plot_logs, write_csv, write_to_readme

if __name__ == '__main__':
    dataset_name = 'MNIST'
    model_name = 'Vanilla_ACNN'

    # HyperParams and Others
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed = 0
    epochs = 1
    batch_size = 64
    learning_rate = 0.01

    graphs = True
    csv = True
    logger_dir = os.path.join(dirname(dirname(abspath(__file__))), 'results')

    download = True     # Doesn't re-download if existent. Def. to True is better
    torch.manual_seed(seed)

    # Loading Data
    data_loc = r'E:\Datasets'
    train_loader, test_loader = load_data(data_loc, batch_size, download=download, dataset=dataset_name)

    # noinspection PyUnresolvedReferences
    model = VanillaACNN(*default_config(dataset=dataset_name, model=model_name)).to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    if not path.exists(logger_dir):
        os.mkdir(logger_dir)
    training_dir, tensorboard_dir = get_directories(model, dataset_name, logger_dir)

    # Tensorboard writer
    if graphs or csv:
        writer = SummaryWriter(tensorboard_dir)  # Handles the creation of tensorboard logs
        train_logger = (np.array([]), np.array([]), np.array([]))  # Handles the creation of png, csv logs
        test_logger = (np.array([]), np.array([]), np.array([]))
    else:  # Avoid weak warnings
        writer, train_logger, test_logger = None, None, None

    tick = time.time()
    for epoch in range(1, epochs + 1):
        train_logger = train(model, device,  # Train Loop
                             train_loader,
                             optimizer, epoch,
                             log_interval=100, writer=writer, logger=train_logger)

        test_logger = test(model, device,  # Evaluation Loop
                           test_loader, epoch,
                           writer=writer, logger=test_logger)
    run_time = time.time() - tick

    if graphs:  # Plot log to graphs
        plot_logs(train_logger, training_dir)
        plot_logs(test_logger, training_dir, test=True)

    if csv:  # Save logs to csv
        write_csv(train_logger, training_dir)
        write_csv(test_logger, training_dir, test=True)

    write_to_readme(batch_size, learning_rate, seed,  # write to Readme.md
                    epochs, run_time, training_dir)
