import os
import shelve
import numpy as np
import copy
import torch
import matplotlib.pyplot as plt

import src.config as config

DEBUG = False


def plot_class_distribution(client_dataloaders, selected_classes):
    
    num_clients = len(client_dataloaders)
    num_selected_classes = len(selected_classes)
    class_counts = {class_label: np.zeros(num_clients) for class_label in selected_classes}

    for client_id, dataloader in enumerate(client_dataloaders):
        for _, labels in dataloader:
            labels_np = labels.numpy() 
            for class_label in selected_classes:
                class_counts[class_label][client_id] += np.sum(labels_np == class_label)

    clients = [f'Client{idx+1}' for idx in range(num_clients)]
    bottom = np.zeros(num_clients)
    colors = plt.cm.get_cmap('viridis', num_selected_classes)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # for idx, class_label in enumerate(selected_classes):
    #     ax.bar(clients, class_counts[class_label], bottom=bottom,
    #            label=f'Class {class_label}', color=colors(idx))
    #     bottom += class_counts[class_label]

    # ax.set_ylabel('Count')
    # ax.set_title('Class distribution across clients')
    # ax.legend(title="Classes")
    #plt.savefig("./class_distribution.png")

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def prepare_tensor_for_plotting(tensor):
    np_image = tensor.cpu().numpy()
    np_image = np.transpose(np_image, (1, 2, 0))
    if np_image.min() < 0 or np_image.max() > 1:
        np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())
    return np_image


def gen_checkpoint_path(alpha: float, threshold: int, method: str, epoch: int, prefix=''):
    file_name = f"{prefix}seed_{config.seed}_alpha_{alpha:.2f}_th_{threshold}_method_{method}_epoch_{epoch}.pth"
    return os.path.join(config.checkpoint_dir, file_name)


def gen_idlg_experiment_path(prefix=''):
    file_name = f"{prefix}seed_{config.seed}"
    return os.path.join(config.idlg_experiment_dir, file_name)


def save_checkpoint(model, optimizer, loss, epoch, alpha, thres, method, prefix=''):
    # Create directory if it doesn't exist
    # Save model state, optimizer state, and additional parameters
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, gen_checkpoint_path(alpha, thres, method, epoch, prefix=prefix))


def load_checkpoint(seed: int, alpha: float, epoch: int, thres=95, method='prune', device='cpu', prefix=''):
    # Load model state, optimizer state, and epoch
    fname = gen_checkpoint_path(alpha, thres, method, epoch, prefix=prefix)
    checkpoint = torch.load(fname, map_location=device)

    # Initialize model and optimizer
    model = config.model
    model.to(device)

    optimizer = config.optimizer

    # Load the state dictionaries
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # Set model to train mode
    model.train()

    return model, optimizer, loss, epoch


def save_idlg_exp_settings(idlg_data, prefix=''):
    # Create directory if it doesn't exist
    with shelve.open(gen_idlg_experiment_path(prefix=prefix)) as db:
        db['idlg_data'] = idlg_data


def load_idlg_exp_settings(prefix=''):
    import time, random, os, pickle, io

    class _CpuUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            return super().find_class(module, name)

    pkl_path = gen_idlg_experiment_path(prefix=prefix) + '.pkl'
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    for attempt in range(10):
        try:
            with shelve.open(gen_idlg_experiment_path(prefix=prefix), flag='r') as db:
                try:
                    return db['idlg_data']
                except RuntimeError:
                    # CUDA tensors on a CPU-only node — remap to CPU
                    raw = db.dict['idlg_data'.encode(db.keyencoding)]
                    return _CpuUnpickler(io.BytesIO(raw)).load()
        except Exception:
            if attempt < 9:
                time.sleep(1 + random.random() * attempt)
            else:
                raise


def prune_gradients(gradients, thres, alpha):
    flattened_gradients = np.concatenate([grads.abs().detach().cpu().numpy().flatten() for grads in gradients])
    threshold = np.percentile(flattened_gradients, thres)

    # Apply pruning to the gradients
    for grads in gradients:
        grad_data = grads.data
        grad_above_thresh = grad_data.abs() > threshold
        grad_data[grad_above_thresh] *= alpha

    if DEBUG:
        flattened_gradients_ = np.concatenate([grads.abs().detach().cpu().numpy().flatten() for grads in gradients])
        assert np.sum(flattened_gradients_) < np.sum(flattened_gradients), "Pruning failed: gradients not reduced"

    return gradients

def clip_gradients(gradients, clip_value=1e-3, protect_overflow=False):
    """
    Clipping gradients for all parameters of the model using PyTorch's built-in function.

    Args:
        model (torch.nn.Module): The model whose gradients will be clipped.
        clip_value (float): The maximum allowed norm for the gradients.
    """
    # get coefficients
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grads_with_norm_.html
    gradients_ = []
    for param in gradients:
        gradients_ += list(param.detach().cpu().numpy().flatten())

    if protect_overflow:
        clip_coeff = np.min([1.0, clip_value / (np.linalg.norm(gradients_) + 1e-6)])
    else:
        clip_coeff = clip_value / (np.linalg.norm(gradients_) + 1e-6)

    # apply coefficients
    for param in gradients:
        param.data *= clip_coeff

    return gradients
