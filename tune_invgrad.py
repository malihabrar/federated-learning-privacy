"""
Hyperparameter tuning for InvGrad attack.

Stage 1: tune TV x LR         (restarts=1, iters fixed)
Stage 2: tune iters x restarts (TV and LR fixed from stage 1)


"""

import argparse
import itertools
import os
import shelve

import numpy as np
import torch
import torch.nn as nn

import breaching
import src.config as config
from src.utils import load_checkpoint, load_idlg_exp_settings, prune_gradients
from src.metrics import Metrics
from src.dataset import CIFAR10_MEAN, CIFAR10_STD

# HPC shelves contain GPU tensors; remap to CPU on CPU-only nodes.
_orig_torch_load = torch.load
def _cpu_torch_load(f, *args, **kwargs):
    kwargs.setdefault('map_location', 'cpu')
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _cpu_torch_load

parser = argparse.ArgumentParser()
parser.add_argument("--stage",     type=int,   default=1)
parser.add_argument("--seed",      type=int,   default=10)
parser.add_argument("--n_images",  type=int,   default=3)
parser.add_argument("--tv",        type=float, default=0.01,  help="Stage 2: fixed TV value")
parser.add_argument("--lr",        type=float, default=0.1,   help="Stage 2: fixed LR value")
parser.add_argument("--device",    type=str,   default="cpu")
args = parser.parse_args()

OUT_DIR = "results/invgrad/tuning"
os.makedirs(OUT_DIR, exist_ok=True)

# ── grids ─────────────────────────────────────────────────────────────────────

STAGE1_TV      = [1e-4, 1e-3, 0.01, 0.1]
STAGE1_LR      = [0.01, 0.05, 0.1, 0.5]
STAGE1_ITERS   = 2000
STAGE1_RESTARTS = 1

STAGE2_ITERS   = [2000]
STAGE2_RESTARTS = [1, 3, 5]

IMG_INDICES = list(range(args.n_images))

# ── load model and images ──────────────────────────────────────────────────────

config.alpha      = 1.0
config.seed       = args.seed
config.method     = "prune"
config.model_name = "lenet"
config.reinit_model()

model, _, _, _ = load_checkpoint(args.seed, 1.0, 0, method="prune", prefix="client_1_")
model = model.to(args.device).float()

idlg_settings = load_idlg_exp_settings()

# ── attack helper ──────────────────────────────────────────────────────────────

class _Meta(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def run_attack(gt_data, label, iters, restarts, tv, lr, device):
    import copy
    m = copy.deepcopy(model).to(device).float()
    criterion = nn.CrossEntropyLoss()

    image = gt_data.to(device).float()
    if image.dim() == 3:
        image = image.unsqueeze(0)

    m.eval()
    label_tensor = torch.tensor([label], dtype=torch.long, device=device)
    logits       = m(image)
    loss         = criterion(logits, label_tensor)
    target_grads = [g.detach().clone() for g in torch.autograd.grad(loss, m.parameters())]

    named_grads = {name: grad for (name, _), grad in zip(m.named_parameters(), target_grads)}
    last_2d = None
    for name, grad in named_grads.items():
        if grad.dim() == 2:
            last_2d = grad
    pred_class = int(torch.argmin(torch.sum(last_2d, dim=-1)).item()) if last_2d is not None else label
    label_pred = torch.tensor([pred_class], dtype=torch.long, device=device)

    m.train()
    setup = dict(device=torch.device(device), dtype=torch.float)
    cfg_attack = breaching.get_attack_config(
        attack="invertinggradients",
        overrides=[
            f"optim.max_iterations={iters}",
            f"restarts.num_trials={restarts}",
            f"regularization.total_variation.scale={tv}",
            f"optim.step_size={lr}",
        ],
    )
    attacker = breaching.attacks.prepare_attack(m, criterion, cfg_attack, setup)

    input_shape     = tuple(image.shape)
    server_metadata = _Meta(
        shape=list(input_shape[1:]),
        modality="vision",
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        num_data_points=input_shape[0],
        labels=None,
        local_hyperparams=None,
    )
    server_payload = [dict(
        parameters=list(m.parameters()),
        buffers=list(m.buffers()),
        metadata=server_metadata,
    )]
    shared_metadata = _Meta(
        labels=label_pred,
        num_data_points=input_shape[0],
        local_hyperparams=None,
    )
    shared_data = [dict(gradients=target_grads, buffers=None, metadata=shared_metadata)]

    reconstructed, _ = attacker.reconstruct(server_payload, shared_data, {})
    dummy = reconstructed["data"].to(device)

    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    gt_01  = (image.squeeze(0).cpu() * std + mean).clamp(0, 1)
    rec_01 = (dummy.detach().squeeze(0).cpu() * std + mean).clamp(0, 1)

    metrics = Metrics([gt_01], [rec_01])
    with torch.no_grad():
        psnr = metrics.compute_psnr()
        ssim = metrics.compute_ssim()
        mse  = metrics.compute_mse()

    return psnr, ssim, mse


# ── stage 1: tune TV x LR ─────────────────────────────────────────────────────

if args.stage == 1:
    print(f"\n{'='*60}")
    print(f"Stage 1: TV x LR tuning  (iters={STAGE1_ITERS}, restarts={STAGE1_RESTARTS})")
    print(f"Seed={args.seed}, images={IMG_INDICES}")
    print(f"{'='*60}\n")

    results = {}
    grid = list(itertools.product(STAGE1_TV, STAGE1_LR))
    total = len(grid) * len(IMG_INDICES)
    done  = 0

    for tv, lr in grid:
        psnrs, ssims = [], []
        for img_idx in IMG_INDICES:
            done += 1
            s       = idlg_settings[img_idx]
            gt_data = s["gt_data"]
            label   = int(s["label"].item())

            psnr, ssim, mse = run_attack(
                gt_data, label,
                iters=STAGE1_ITERS, restarts=STAGE1_RESTARTS,
                tv=tv, lr=lr, device=args.device
            )
            psnrs.append(psnr)
            ssims.append(ssim)
            print(f"[{done}/{total}] TV={tv:.0e}  LR={lr}  img={img_idx}  "
                  f"PSNR={psnr:.2f}  SSIM={ssim:.4f}  MSE={mse:.4f}")

        key = (tv, lr)
        results[key] = (np.mean(psnrs), np.mean(ssims))

    print(f"\n{'─'*60}")
    print("Stage 1 summary (sorted by mean PSNR):")
    print(f"{'TV':<10} {'LR':<8} {'PSNR':>8} {'SSIM':>8}")
    print(f"{'─'*60}")
    for (tv, lr), (psnr, ssim) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"{tv:<10.0e} {lr:<8}  {psnr:>7.2f}  {ssim:>7.4f}")

    best_tv, best_lr = max(results, key=lambda k: results[k][0])
    best_psnr = results[(best_tv, best_lr)][0]
    print(f"\nBest: TV={best_tv:.0e}  LR={best_lr}  PSNR={best_psnr:.2f}")
    print(f"\nRun Stage 2 with:")
    print(f"  python tune_invgrad.py --stage 2 --seed {args.seed} "
          f"--n_images {args.n_images} --tv {best_tv} --lr {best_lr} --device {args.device}")

    shelve_path = os.path.join(OUT_DIR, f"stage1_seed_{args.seed}")
    with shelve.open(shelve_path) as d:
        d["results"] = results
        d["best_tv"] = best_tv
        d["best_lr"] = best_lr
    print(f"\nSaved to {shelve_path}")


# ── stage 2: tune iters x restarts ────────────────────────────────────────────

elif args.stage == 2:
    tv = args.tv
    lr = args.lr

    print(f"\n{'='*60}")
    print(f"Stage 2: iters x restarts tuning  (TV={tv}, LR={lr})")
    print(f"Seed={args.seed}, images={IMG_INDICES}")
    print(f"{'='*60}\n")

    results = {}
    grid  = list(itertools.product(STAGE2_ITERS, STAGE2_RESTARTS))
    total = len(grid) * len(IMG_INDICES)
    done  = 0

    for iters, restarts in grid:
        psnrs, ssims = [], []
        for img_idx in IMG_INDICES:
            done += 1
            s       = idlg_settings[img_idx]
            gt_data = s["gt_data"]
            label   = int(s["label"].item())

            psnr, ssim, mse = run_attack(
                gt_data, label,
                iters=iters, restarts=restarts,
                tv=tv, lr=lr, device=args.device
            )
            psnrs.append(psnr)
            ssims.append(ssim)
            print(f"[{done}/{total}] iters={iters}  restarts={restarts}  img={img_idx}  "
                  f"PSNR={psnr:.2f}  SSIM={ssim:.4f}  MSE={mse:.4f}")

        key = (iters, restarts)
        results[key] = (np.mean(psnrs), np.mean(ssims))

    print(f"\n{'─'*60}")
    print("Stage 2 summary (sorted by mean PSNR):")
    print(f"{'Iters':<8} {'Restarts':<10} {'PSNR':>8} {'SSIM':>8}")
    print(f"{'─'*60}")
    for (iters, restarts), (psnr, ssim) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"{iters:<8} {restarts:<10}  {psnr:>7.2f}  {ssim:>7.4f}")

    best_iters, best_restarts = max(results, key=lambda k: results[k][0])
    best_psnr = results[(best_iters, best_restarts)][0]
    print(f"\nBest: iters={best_iters}  restarts={best_restarts}  PSNR={best_psnr:.2f}")
    print(f"\nFinal hyperparameters:")
    print(f"  TV={tv}  LR={lr}  iters={best_iters}  restarts={best_restarts}")

    shelve_path = os.path.join(OUT_DIR, f"stage2_seed_{args.seed}_tv_{tv}_lr_{lr}")
    with shelve.open(shelve_path) as d:
        d["results"]       = results
        d["best_iters"]    = best_iters
        d["best_restarts"] = best_restarts
        d["tv"]            = tv
        d["lr"]            = lr
    print(f"\nSaved to {shelve_path}")
