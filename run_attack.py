"""
Unified entry point for gradient-leakage attacks.

Usage:
    python run_attack.py --attack invgrad --model vit  --method prune --alpha 0.95 --seed 10 --img_inx 0
    python run_attack.py --attack idlg    --model lenet --method clip  --alpha 1.0  --seed 10 --img_inx 0

Supported combinations:
    --attack  : invgrad | idlg
    --model   : lenet   | vit
    --method  : prune   | clip
    --alpha   : 0.0 – 1.0  (defence strength; 1.0 = no defence)
    --device  : cpu     | cuda  (default: cuda if available, else cpu)

Notes:
    - A trained model checkpoint must exist in results/checkpoints/.
      Train one first with: python nn_train.py --model <model> --method <method> --alpha <alpha> --seed <seed>
    - iDLG is only evaluated on LeNet in this project (L-BFGS is impractical for ViT-scale models).
"""
import os
import argparse
import random
import shelve

import numpy as np
import src.config as config
from src.utils import load_checkpoint, load_idlg_exp_settings

parser = argparse.ArgumentParser(description="Run a gradient-leakage attack.")
parser.add_argument("--attack",   type=str,   required=True,
                    choices=["invgrad", "idlg"], help="Attack type.")
parser.add_argument("--model",    type=str,   default="lenet",
                    choices=["lenet", "vit"],   help="Model architecture.")
parser.add_argument("--method",   type=str,   default="prune",
                    choices=["prune", "clip"],  help="Defence method.")
parser.add_argument("--alpha",    type=lambda x: round(float(x), 2), default=1.0,
                    help="Defence strength (1.0 = no defence, 0.0 = strict LGP).")
parser.add_argument("--seed",     type=int,   default=10)
parser.add_argument("--img_inx",  type=int,   default=0,  help="Image index (0–29).")
parser.add_argument("--iters",    type=int,   default=None,
                    help="Optimiser iterations (default: 300 for iDLG, 2000 for InvGrad).")
parser.add_argument("--restarts", type=int,   default=None,
                    help="Random restarts (default: 5 for iDLG, 3 for InvGrad).")
parser.add_argument("--device",   type=str,   default=None,
                    help="'cpu' or 'cuda'. Defaults to cuda if available.")

args = parser.parse_args()

import torch

# Fix seeds so attack initialisation is reproducible given the same --seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.attack == "idlg" and args.model == "vit":
    raise ValueError(
        "iDLG is not supported for ViT. L-BFGS is computationally infeasible for "
        "86M-parameter models. Use --attack invgrad instead."
    )

if args.device is None:
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

# Apply defaults per attack
if args.attack == "idlg":
    if args.iters    is None: args.iters    = 300
    if args.restarts is None: args.restarts = 5
else:
    if args.iters    is None: args.iters    = 2000
    if args.restarts is None: args.restarts = 3

# Configure global state
config.alpha      = args.alpha
config.seed       = args.seed
config.method     = args.method
config.model_name = args.model
config.reinit_model()

ckpt_prefix = "client_1_vit_" if args.model == "vit" else "client_1_"
from src.utils import gen_checkpoint_path
ckpt_path = gen_checkpoint_path(config.alpha, 95, config.method, 0, prefix=ckpt_prefix)
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
        f"No checkpoint found at: {ckpt_path}\n"
        f"Train the model first with:\n"
        f"  python nn_train.py --model {args.model} --method {args.method} "
        f"--alpha {args.alpha} --seed {args.seed}"
    )
model, optimizer, loss, epoch = load_checkpoint(
    config.seed, config.alpha, 0, method=config.method, prefix=ckpt_prefix,
    device=args.device,
)

print(f"Attack  : {args.attack}")
print(f"Model   : {args.model}  |  Defence: {args.method}  alpha={args.alpha}  seed={args.seed}")
print(f"Image   : {args.img_inx}  |  Iters: {args.iters}  Restarts: {args.restarts}  Device: {args.device}")

# InvGrad 
if args.attack == "invgrad":
    from src.inverting_gradients import InvGrad

    idlg_settings_all = load_idlg_exp_settings()
    settings  = idlg_settings_all[args.img_inx]
    gt_data   = settings["gt_data"]
    label     = int(settings["label"].item())

    exp_name = (f"img_{args.img_inx}_seed_{config.seed}_alpha_{config.alpha:.2f}"
                f"_method_{args.method}_model_{args.model}"
                f"_iters_{args.iters}_restarts_{args.restarts}")
    out_dir     = os.path.join(config.results_dir, "invgrad")
    shelve_file = os.path.join(out_dir, "bin", exp_name)
    figpath     = os.path.join(out_dir, "figs", exp_name + ".png")

    attacker = InvGrad(
        model=model, gt_data=gt_data, label=label,
        device=args.device, iters=args.iters, restarts=args.restarts,
        tv=0.001, alpha=config.alpha, thres=95, prune_method=config.method,
    )
    dummy_data, label_pred, mse_val, psnr_val, ssim_val = attacker.attack()

    os.makedirs(os.path.dirname(figpath),     exist_ok=True)
    os.makedirs(os.path.dirname(shelve_file), exist_ok=True)
    attacker.save_reconstruction(dummy_data, mse_val, psnr_val, ssim_val, out=figpath)

    d = shelve.open(shelve_file)
    d["psnr"] = psnr_val; d["ssim"] = ssim_val; d["mse"] = mse_val
    d["label_pred"] = label_pred.item(); d["label_true"] = label
    d.close()

    print(f"Done — PSNR={psnr_val:.2f}  SSIM={ssim_val:.4f}  MSE={mse_val:.6f}")
    print(f"Saved to {shelve_file}")

# iDLG
else:
    from src.idlg_modified import idlg_experiment

    exp_name    = (f"img_{args.img_inx}_seed_{config.seed}_alpha_{config.alpha:.2f}"
                   f"_method_{args.method}_{args.restarts}")
    shelve_file = os.path.join(config.idlg_experiment_dir, "bin", exp_name)
    figpath     = os.path.join(config.idlg_experiment_dir, "figs", exp_name + ".png")

    psnr_vals, ssim_vals, img_reconstructed = idlg_experiment(
        model, img_inx=args.img_inx, iteration=args.restarts,
        alpha=config.alpha, thres=95, prune_method=config.method,
        figpath=figpath, device=args.device, load_experiment=True,
    )

    os.makedirs(os.path.dirname(shelve_file), exist_ok=True)
    d = shelve.open(shelve_file)
    d["psnr"] = psnr_vals; d["ssim"] = ssim_vals; d["img_reconsctructed"] = img_reconstructed
    d.close()

    print(f"Done — final PSNR={psnr_vals[-1]:.2f}  SSIM={ssim_vals[-1]:.4f}")
    print(f"Saved to {shelve_file}")
