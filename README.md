# Gradient Leakage in Federated Learning: Attacks and Defences

This repository accompanies a BSc thesis at DTU evaluating privacy attacks and defences in federated learning on CIFAR-10. It provides a parametrizable evaluation pipeline for comparing gradient-leakage attacks across model architectures and defence mechanisms.

## What's included

**Defences**
- No defence (baseline)
- Gradient clipping
- Strict LGP - α=0 (full gradient zeroing)
- PLGP - Proportional Large Gradient Pruning (α ∈ [0, 1])

**Attacks**
- iDLG (Zhao et al., 2020) - evaluated on LeNet
- Inverting Gradients (Geiping et al., 2020) - evaluated on LeNet and ViT

**Architectures**
- LeNet (CIFAR-10, trained from scratch)
- ViT-B/16 (fine-tuned from `nielsr/vit-base-patch16-224-in21k-finetuned-cifar10`)

---

## Installation

```bash
pip install -r requirements.txt
```

For GPU support (recommended for ViT):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## Usage

### 1. Train a model

```bash
python nn_train.py --model lenet --method prune --alpha 0.95 --seed 10
python nn_train.py --model vit   --method clip  --alpha 1.0  --seed 10
```

- `--model`  : `lenet` or `vit`
- `--method` : `prune` (PLGP/LGP) or `clip` (gradient clipping)
- `--alpha`  : defence strength - `1.0` = no defence, `0.0` = strict LGP, values in between = PLGP
- `--seed`   : random seed for reproducibility

Checkpoints are saved to `results/checkpoints/`.

### 2. Run an attack

```bash
python run_attack.py --attack invgrad --model vit   --method prune --alpha 0.95 --seed 10 --img_inx 0
python run_attack.py --attack idlg    --model lenet --method prune --alpha 1.0  --seed 10 --img_inx 0
```

- `--attack`   : `invgrad` or `idlg`
- `--model`    : `lenet` or `vit`
- `--method`   : `prune` or `clip`
- `--alpha`    : must match a trained checkpoint
- `--img_inx`  : image index (0–29)
- `--iters`    : optimiser iterations (default: 2000 for InvGrad, 300 for iDLG)
- `--restarts` : random restarts (default: 3 for InvGrad, 5 for iDLG)
- `--device`   : `cpu` or `cuda` (defaults to cuda if available)

Results (PSNR, SSIM, MSE, reconstructed image) are saved to `results/invgrad/` or `results/idlg_experiments/`.

> **Note:** iDLG on ViT is not supported - L-BFGS is computationally infeasible for 86M-parameter models. Use `--attack invgrad` for ViT. ViT also requires a CUDA-capable GPU due to Flash Attention's backward pass.

---

## Results and analysis

After running experiments, aggregate results with:

```bash
python results_summary.py
```

This produces `results/summary.csv` and `results/summary_table.txt`.

Interactive analysis and figures are available in the notebooks:
- `notebook_accuracy.ipynb` - training accuracy curves
- `notebook_idlg.ipynb` - iDLG PSNR/SSIM convergence
- `notebook_attack_analysis.ipynb` - iDLG PSNR distributions and convergence rates
- `notebook_invgrad.ipynb` - InvGrad PSNR/SSIM distributions and label inference

---

## Project structure

```
run_attack.py          # Unified attack entry point
nn_train.py            # Federated learning training
results_summary.py     # Aggregate results into table and CSV
tune_invgrad.py        # InvGrad hyperparameter grid search
src/
  vit.py               # ViT wrapper (HuggingFace)
  lenet.py             # LeNet architecture
  inverting_gradients.py  # InvGrad attack
  idlg_modified.py     # iDLG attack
  utils.py             # Gradient pruning, clipping, checkpointing
  config.py            # Global configuration
  prepare.py           # Dataset and model initialisation
scripts/               # HPC batch job schedulers (LSF/bsub)
```
