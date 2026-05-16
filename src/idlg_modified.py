import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from tqdm import trange

from src.metrics import Metrics
from src.utils import prepare_tensor_for_plotting, load_idlg_exp_settings, clip_gradients
from src.dataset import CIFAR10_MEAN, CIFAR10_STD


DEBUG = False


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


class iDLG:
    def __init__(
        self,
        model,
        orig_img,
        gt_data,
        label,
        device,
        alpha=1, 
        thres=95, # alpha_threshold=95
        prune=False,
        prune_method="prune",
        clip_value=1e-3,
    ) -> None:
        device = 'cpu'

        self.alpha = alpha
        self.thres = thres
        self.prune = prune
        self.prune_method = prune_method
        self.clip_value = clip_value
        # HACK, need to adjust API such that prune flag comes is not used
        if prune_method == 'clip':
            self.prune = False

        self.orig_img = orig_img

        if device == 'cpu':
            self.device = device
            self.dtype = torch.float64
            self.model = model.to(self.device).double()
            self.orig_img = orig_img.to(self.device).double()
            self.gt_data = gt_data.to(self.device).double()
            self.criterion = nn.CrossEntropyLoss().to(self.device).double()
            self.label = label.to(self.device)
        elif device == 'gpu-64':
            self.device = 'cuda'
            self.dtype = torch.float64
            self.model = model.to(self.device).double()
            self.orig_img = orig_img.double()
            self.gt_data = gt_data.to(self.device).double()
            self.criterion = nn.CrossEntropyLoss().to(self.device).double()
            self.label = label.to(self.device)
        else:
            self.device = device
            self.dtype = torch.float32
            self.model = model.to(self.device)
            self.gt_data = gt_data.to(self.device)
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            self.label = label.to(self.device)
        self.tp = transforms.ToTensor()
        self.tt = transforms.ToPILImage()
        self.abs_tol = 1e-10
        # for denormalizing to [0,1] before metric computation
        self._mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std  = torch.tensor(CIFAR10_STD,  dtype=torch.float32).view(3, 1, 1)

    @staticmethod
    def label_to_onehot(target, num_classes=10):
        target = torch.unsqueeze(target, 1)
        onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
        onehot_target.scatter_(1, target, 1)
        return onehot_target

    def attack(self, iteration=0):
        gt_onehot_label = self.label_to_onehot(self.label)
        iteration = 300
        #iDLG training image reconstruction:
        self.model.eval()

        predicted = self.model(self.gt_data)
        loss = self.criterion(predicted, gt_onehot_label)
        dy_dx = torch.autograd.grad(loss, self.model.parameters())
        orig_dy_dx = list((_.detach().clone() for _ in dy_dx))
        if self.prune:
            # Prune gradients
            orig_dy_dx = prune_gradients(orig_dy_dx, thres=self.thres, alpha=self.alpha)
        
        if self.prune_method == 'clip':
            # Clip gradients
        
            orig_dy_dx = clip_gradients(orig_dy_dx, self.clip_value)

        # initialize dummy in the correct iteration, respecting the random seed
        for i in range(iteration):
            dummy_data = torch.randn(self.gt_data.size(), dtype=self.dtype).to(self.device).requires_grad_(True)

        # init with ground truth:
        # dummy_data = self.gt_data.clone().to(self.device).requires_grad_(True)
        label_pred = torch.argmin(torch.sum(orig_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
        optimizer = torch.optim.LBFGS([dummy_data], lr=1.0, max_iter=50,
                              tolerance_grad=1e-09, tolerance_change=1e-11,
                              history_size=100, line_search_fn='strong_wolfe')

        history = []
        losses = []
        ssim_vals = []
        psnr_vals = []
        mse_vals = []

        for iters in trange(iteration):
            def closure():
                optimizer.zero_grad()
                dummy_pred = self.model(dummy_data)
                dummy_loss = self.criterion(dummy_pred, label_pred)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, orig_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)

            current_loss = closure()
            losses.append(current_loss)
            # Denormalize to [0,1] for correct PIL conversion and metrics
            recon_01 = (dummy_data[0].detach().float().cpu() * self._std + self._mean).clamp(0, 1)
            orig_01  = (self.orig_img.float().cpu()           * self._std + self._mean).clamp(0, 1)
            history.append(self.tt(recon_01))
            metrics = Metrics(ground_truth_imgs=[orig_01], reconstructed_imgs=[recon_01])
            ssim = metrics.compute_ssim()
            psnr = metrics.compute_psnr()
            mse = metrics.compute_mse()
            ssim_vals.append(ssim)
            psnr_vals.append(psnr)
            mse_vals.append(mse)

            if DEBUG:
                np_image = self.orig_img.cpu().numpy()
                np_image = np.transpose(np_image, (1, 2, 0))
                if np_image.min() < 0 or np_image.max() > 1:
                    np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())

                losses_arr = [t.detach().cpu().item() for t in losses]
                losses_arr = np.array(losses_arr)

                fig = plt.figure("debug", figsize=(8, 6))
                fig.clear()

                gs = GridSpec(2, 2, figure=fig)
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.clear()
                ax1.imshow(np_image)
                ax1.set_title(f"Org Image")
                ax1.axis("off")

                ax2 = fig.add_subplot(gs[0, 1])
                ax2.clear()
                ax2.imshow(history[-1])
                ax2.set_title(f"Reconstructed Image")
                ax2.axis("off")

                ax_loss = fig.add_subplot(gs[1, :])  # Bottom spanning both columns
                ax_loss.clear()
                ax_loss.plot(np.log10(losses_arr), label='Loss')
                ax_loss.set_xlabel('Ite [n]')
                ax_loss.set_ylabel('Loss [log10]')
                ax_loss.grid()

                fig.suptitle(f"Iteration {iters} - Loss: {current_loss:.2e} - PSNR: {psnr:.2f} - SSIM: {ssim:.3f} - MSE: {mse:.3f}")
                plt.show()
                plt.draw()
                plt.pause(0.01)

        return dummy_data, label_pred, history, losses, ssim_vals, psnr_vals, mse_vals


def idlg_experiment(model, img_inx=0, iteration=0, alpha=0.25, thres=95, prune_method='prune', figpath='', device='cpu', load_experiment=True):
    idlg_settings = {}
    if load_experiment:
        idlg_settings = load_idlg_exp_settings()

    idlg_settings = idlg_settings[img_inx]

    prune = False
    if prune_method == 'prune' and alpha < 1:
        prune = True

    idlg = iDLG(model=model, alpha=alpha, thres=thres, prune=prune, prune_method=prune_method, device=device, **idlg_settings)
    dummy_data, label_pred, history, losses, ssim_vals, psnr_vals, mse_vals = idlg.attack(iteration=iteration)

    img_reconstructed = history[-1]
    if figpath != '':
        fig_dir = os.path.dirname(figpath)
        os.makedirs(fig_dir, exist_ok=True)

        losses_arr = [t.detach().cpu().item() for t in losses]
        losses_arr = np.array(losses_arr)

        fig = plt.figure(figsize=(8, 6))
        fig.suptitle(f"Iterations {len(losses_arr)} - Loss: {losses_arr[-1]:.2e} - PSNR: {psnr_vals[-1]:.2f} - SSIM: {ssim_vals[-1]:.3f} - MSE: {mse_vals[-1]:.3f}")

        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(prepare_tensor_for_plotting(idlg_settings['orig_img']))
        ax1.set_title(f"Org Image")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        # ax2.imshow(prepare_tensor_for_plotting(dummy_data.detach()[0]))
        ax2.imshow(img_reconstructed)
        ax2.set_title(f"Reconstructed Image")
        ax2.axis("off")

        ax_loss = fig.add_subplot(gs[1, :])  # Bottom spanning both columns
        ax_loss.plot(np.log10(losses_arr), label='Loss')
        ax_loss.set_xlabel('Ite [n]')
        ax_loss.set_ylabel('Loss [log10]')
        ax_loss.grid()
        fig.savefig(figpath, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return psnr_vals, ssim_vals, img_reconstructed
