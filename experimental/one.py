import os
from os import path

def get_directories(model, dataset, parent_dir):
	"""
    Returns paths to save training logs
    :param model: Instance of the model class
    :param dataset: String containing dataset name
    :param parent_dir: dir where sub directories are to be made
    :return: Training_dir and Tensorboard_dir, for saving required files
    """
    if dataset not in ['MNIST', 'CIFAR10', 'SVHN']:
    raise NotImplementedError('Only MNIST/SVHN/CIFAR10')

    model_name = model.__class__.__name__    
    parent_dir = os.path.abspath(parent_dir)
    dataset_dir = os.path.join(parent_dir, dataset)
    model_dir = os.path.join(dataset_dir, model_name)

    if not path.exists(dataset_dir):
        os.mkdir(dataset_dir)
 
    if not path.exists(model_dir):
        os.mkdir(model_dir)

    i = 1
    while(True):
        training_dir = os.path.join(model_dir, f'Training_{i}')
        if not path.exists(training_dir):
            os.mkdir(training_dir)
            break
        i += 1
    
    tensorboard_dir = os.path.join(training_dir, 'Tensorboard_Summary')
    return training_dir, tensorboard_dir