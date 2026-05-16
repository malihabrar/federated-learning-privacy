import random

import numpy as np
import torch
import torch.nn as nn

from src.dataset import CIFARDataset, iid_dataloader, filter_dataset_by_class
from src.lenet import weights_init
from src.utils import plot_class_distribution

import src.config as config


def prepare(seed=None):
    if seed is None:
        seed = config.seed
    else:
        config.seed = seed

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Intialize model:
    global_model = config.model.to(device)
    if config.model_name != "vit":
        global_model.apply(weights_init)

    all_classes = list(range(10))
    criterion = nn.CrossEntropyLoss().to(device)

    batch_size = 50
    num_clients = 5
    cifar_data = CIFARDataset(batch_size=batch_size, num_clients=num_clients)
    train_dataset, validation_dataset = cifar_data.get_dataset()

    iid_client_train_loader = iid_dataloader(train_dataset, batch_size=batch_size, num_clients=num_clients)
    plot_class_distribution(iid_client_train_loader, all_classes)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    num_images_per_class = 3
    clients_dataset = {}

    for index, client_loader in enumerate(iid_client_train_loader):
        client_dataset = client_loader.dataset
        clients_dataset[index] = filter_dataset_by_class(client_dataset, all_classes, num_images_per_class)

    return iid_client_train_loader, device, criterion, validation_loader, train_loader, clients_dataset, global_model
