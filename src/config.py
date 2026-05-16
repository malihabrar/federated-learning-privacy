import os
import torch
from src.lenet import LeNet

seeds = [10, 20, 30]
seed = seeds[0]

alphas = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 1.0]
alpha = alphas[0]

methods = ["prune", "clip"]
method = methods[0]  # prune or clip

model_name = "lenet"  # "lenet" or "vit"

IDLG_N_images = 30  # 3 per class × 10 classes
IDLG_retries = 15

# Intialize model:
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_model():
    if model_name == "vit":
        from src.vit import ViTWrapper
        m = ViTWrapper().to(device)
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    else:
        m = LeNet().to(device)
        opt = torch.optim.Adam(m.parameters())
    return m, opt

model, optimizer = make_model()


def reinit_model():
    """Call after changing config.model_name to recreate model and optimizer."""
    global model, optimizer
    model, optimizer = make_model()

build_dir = "build/"
os.makedirs(build_dir, exist_ok=True)

results_dir = "results/"
os.makedirs(results_dir, exist_ok=True)

checkpoint_dir = os.path.join(results_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

idlg_experiment_dir = os.path.join(results_dir, "idlg_experiments")
os.makedirs(idlg_experiment_dir, exist_ok=True)
