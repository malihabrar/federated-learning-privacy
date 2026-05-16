"""
Inverting Gradients attack (Geiping et al., 2020) via the breaching library.
Uses Adam optimisation with cosine-similarity gradient matching and total variation regularisation.
Tracks PSNR / SSIM / MSE for comparison against iDLG.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from torch import nn

import breaching

from src.metrics import Metrics
from src.utils import prune_gradients, clip_gradients
from src.dataset import CIFAR10_MEAN, CIFAR10_STD


class _Meta(dict):
    """Supports both attribute (meta.key) and dict (meta["key"]) access.


    """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class InvGrad:
    def __init__(
        self,
        model:        nn.Module,
        gt_data:      torch.Tensor,
        label:        int,
        device:       str   = "cpu",
        iters:        int   = 2000,
        restarts:     int   = 3,
        tv:           float = 0.001,
        lr:           float = 0.5,
        alpha:        float = 1.0,
        thres:        int   = 95,
        prune_method: str   = "prune",
        clip_value:   float = 1e-3,
    ) -> None:
        """
        Args:
            model:        trained model (LeNet or ViT)
            gt_data:      single CIFAR-10 image, shape (3, 32, 32) or (1, 3, 32, 32),
                          normalized (mean/std applied) — as stored by the training checkpoint
            label:        integer class index (0-9)
            device:       "cpu" or "cuda"
            iters:        Adam iterations (paper uses 24000; 4000 for quick runs)
            restarts:     random restarts (paper uses 8 for best results)
            tv:           total variation regularisation scale
            lr:           Adam step size (default 0.5, tuned on CIFAR-10 LeNet)
            alpha:        pruning scale factor — 1.0 means no pruning defence
            thres:        percentile threshold for pruning (default 95)
            prune_method: "prune", "clip", or "none"
            clip_value:   gradient norm bound used when prune_method="clip"
        """
        self.device       = device
        self.model        = model.to(device).float()
        self.label        = label
        self.iters        = iters
        self.restarts     = restarts
        self.tv           = tv
        self.lr           = lr
        self.alpha        = alpha
        self.thres        = thres
        self.prune_method = prune_method
        self.clip_value   = clip_value
        self.criterion    = nn.CrossEntropyLoss()

        if gt_data.dim() == 3:
            gt_data = gt_data.unsqueeze(0)
        self.gt_data = gt_data.to(device).float()

    @staticmethod
    def extract_label(named_grads: dict) -> int:
        """iDLG label extraction: argmin of row-sums of the last linear layer's weight gradient.

        Works for any model — finds the last parameter whose name ends with
        '.weight' (or equals 'fc.weight') and has a 2-D gradient, which is
        the classifier weight regardless of architecture (LeNet, ViT, etc.).
        """
        classifier_grad = None
        for name, grad in named_grads.items():
            if name.endswith(".weight") or name == "fc.weight":
                if grad.dim() == 2:
                    classifier_grad = grad
        if classifier_grad is None:
            raise ValueError("Could not find a 2-D weight gradient for label extraction.")
        return int(torch.argmin(torch.sum(classifier_grad, dim=-1)).item())

    def attack(self):
        """
        Run the Inverting Gradients attack.

        Returns:
            dummy_data: reconstructed image tensor, shape (1, 3, 32, 32), normalized space
            label_pred: predicted label tensor, shape (1,)
            mse_val:    final MSE
            psnr_val:   final PSNR (dB)
            ssim_val:   final SSIM
        """
        image = self.gt_data

        # Capture gradients — model.eval(), single forward/backward, same as iDLG
        self.model.eval()
        label_tensor = torch.tensor([self.label], dtype=torch.long, device=self.device)
        logits       = self.model(image)
        loss         = self.criterion(logits, label_tensor)
        target_grads = list(torch.autograd.grad(loss, self.model.parameters()))
        target_grads = [g.detach().clone() for g in target_grads]

        if self.prune_method == "prune" and self.alpha < 1.0:
            target_grads = prune_gradients(target_grads, thres=self.thres, alpha=self.alpha)
        elif self.prune_method == "clip":
            target_grads = clip_gradients(target_grads, self.clip_value)

        named_grads = {name: grad for (name, _), grad in zip(self.model.named_parameters(), target_grads)}
        pred_class  = self.extract_label(named_grads)
        label_pred = torch.tensor([pred_class], dtype=torch.long, device=self.device)
        print(f"  [InvGrad] Label: predicted={pred_class}, ground_truth={self.label} "
              f"({'correct' if pred_class == self.label else 'WRONG'})")

        # breaching requires model in train mode during reconstruction
        self.model.train()

        setup      = dict(device=torch.device(self.device), dtype=torch.float)
        cfg_attack = breaching.get_attack_config(
            attack="invertinggradients",
            overrides=[f"optim.max_iterations={self.iters}",
                       f"restarts.num_trials={self.restarts}",
                       f"regularization.total_variation.scale={self.tv}",
                       f"optim.step_size={self.lr}"],
        )
        attacker = breaching.attacks.prepare_attack(self.model, self.criterion, cfg_attack, setup)

        # _Meta supports both meta.key (dot) and meta["key"] (bracket) access,
        # which breaching uses interchangeably across server_payload and shared_data metadata
        input_shape     = tuple(image.shape)   # (1, 3, 32, 32)
        server_metadata = _Meta(
            shape=list(input_shape[1:]),        # [3, 32, 32]
            modality="vision",
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD,
            num_data_points=input_shape[0],
            labels=None,
            local_hyperparams=None,
        )
        server_payload = [dict(
            parameters=list(self.model.parameters()),
            buffers=list(self.model.buffers()),
            metadata=server_metadata,
        )]

        shared_metadata = _Meta(
            labels=label_pred,
            num_data_points=input_shape[0],
            local_hyperparams=None,
        )
        shared_data = [dict(
            gradients=target_grads,
            buffers=None,
            metadata=shared_metadata,
        )]

        reconstructed, stats = attacker.reconstruct(server_payload, shared_data, {})
        dummy_data  = reconstructed["data"].to(self.device)
        self.stats  = stats   # kept for save_reconstruction()

        mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
        std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
        gt_01  = (image.squeeze(0).cpu() * std + mean).clamp(0, 1)
        rec_01 = (dummy_data.detach().squeeze(0).cpu() * std + mean).clamp(0, 1)

        metrics = Metrics([gt_01], [rec_01])
        with torch.no_grad():
            mse_val  = metrics.compute_mse()
            psnr_val = metrics.compute_psnr()
            ssim_val = metrics.compute_ssim()

        print(f"  Final  PSNR={psnr_val:.2f}dB  SSIM={ssim_val:.4f}  MSE={mse_val:.6f}")

        return dummy_data, label_pred, mse_val, psnr_val, ssim_val

    def save_reconstruction(
        self,
        dummy_data: torch.Tensor,
        mse_val:   float,
        psnr_val:  float,
        ssim_val:  float,
        out: str = "results/attacks/invgrad_reconstruction.png",
    ) -> None:
        """Save a figure matching iDLG's layout: images on top, loss curve below."""
        out_dir = os.path.dirname(out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        from src.utils import prepare_tensor_for_plotting

        image = self.gt_data

        # collect per-iteration loss values across all restarts
        loss_vals = []
        for key, val in self.stats.items():
            if key.startswith("Trial_") and key.endswith("_Val"):
                loss_vals.extend(val)
        loss_arr = np.array(loss_vals)

        fig = plt.figure(figsize=(10, 6))
        fig.suptitle(
            f"Iterations {len(loss_arr)} — "
            f"PSNR: {psnr_val:.2f} dB  SSIM: {ssim_val:.4f}  MSE: {mse_val:.6f}"
        )
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(prepare_tensor_for_plotting(image.squeeze(0).cpu()))
        ax1.set_title("Ground Truth")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(prepare_tensor_for_plotting(dummy_data.detach().squeeze(0).cpu()))
        ax2.set_title("Reconstruction")
        ax2.axis("off")

        ax_loss = fig.add_subplot(gs[1, :])
        if loss_arr.size > 0 and np.all(np.isfinite(loss_arr)) and np.all(loss_arr > 0):
            ax_loss.plot(np.log10(loss_arr), label="Cosine loss")
        else:
            ax_loss.plot(loss_arr, label="Cosine loss")
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("Loss (log10)")
        ax_loss.legend()
        ax_loss.grid()

        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Reconstruction saved to {out}")
