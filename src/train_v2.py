import os
import copy

import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np
import torch

from src.train_convex_experiment import ModifiediDLG
from src.utils import prepare_tensor_for_plotting, save_checkpoint, save_idlg_exp_settings, clip_gradients
import src.config as config

DEBUG = False


class FederatedLearning:
    def __init__(
        self,
        model,
        clients_dataloaders,
        num_clients_per_round,
        num_local_epochs,
        lr,
        device,
        criterion,
        filtered_train_dataset,
        train_dataloader,
        test_dataloader,
        max_rounds,
        alpha,
        prune_method="prune",
        clip_value=1e-3,
        alpha_threshold=95
    ) -> None:
        self.global_model = model
        self.clients_dataloader = clients_dataloaders
        self.num_local_epochs = num_local_epochs
        self.lr = lr
        self.device = device
        self.criterion = criterion
        self.filtered_train_dataset = filtered_train_dataset
        self.alpha = alpha
        self.num_clients_per_round = num_clients_per_round
        
        # self.test_dataloader = train_dataloader
        # self.train_dataloader = test_dataloader
        self.test_dataloader = test_dataloader
        self.train_dataloader = train_dataloader
        self.max_rounds = max_rounds
        self.prune_method = prune_method
        self.optimizer = None
        self.cur_loss = None
        self.thres = alpha_threshold  # default threshold for pruning
        self.clip_value = clip_value  # default value for gradient clipping

    def clip_gradients(self, model, clip_value=1e-3):
        """
        Clipping gradients for all parameters of the model using PyTorch's built-in function.

        Args:
            model (torch.nn.Module): The model whose gradients will be clipped.
            clip_value (float): The maximum allowed norm for the gradients.
        """
        # Collect only parameters with gradients.
        if DEBUG:
            parm_grad = [p.grad for p in model.parameters() if p.grad is not None]
            print(f"grad norm before clip {torch.nn.utils.get_total_norm(parm_grad, norm_type=2, foreach=True)}")
            # perform calculation manually

            parm_grad_ = copy.deepcopy(parm_grad)
            parm_grad_ = clip_gradients(parm_grad_, clip_value=clip_value)

        parameters = [p for p in model.parameters() if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(parameters, clip_value, norm_type=2, foreach=True)

        if DEBUG:
            parm_grad = [p.grad for p in model.parameters() if p.grad is not None]
            print(f"grad norm after clip {torch.nn.utils.get_total_norm(parm_grad, norm_type=2, foreach=True)}")

            # compare with manual calculation
            for inx, param in enumerate(parm_grad_):
                assert np.allclose(parm_grad_[inx], parm_grad[inx])

            

    def prune_gradients(self, model, thres, alpha):
        grad_cumsum = 0
        grad_pruned_cumsum = 0

        all_grads = torch.cat([param.grad.data.abs().flatten()
                                for param in model.parameters() if param.grad is not None])
        # Sample for quantile estimation — exact on small models, fast approximation on large ones (e.g. ViT 86M params)
        sample_size = min(200_000, all_grads.numel())
        if sample_size < all_grads.numel():
            idx = torch.randperm(all_grads.numel(), device=all_grads.device)[:sample_size]
            sampled = all_grads[idx]
        else:
            sampled = all_grads
        threshold = torch.quantile(sampled, thres / 100.0).item()

        for param in model.parameters():
            if param.grad is not None:
                if DEBUG:
                    grad_cumsum += param.grad.data.abs().sum()

                grad_above_thresh = param.grad.data.abs() > threshold
                param.grad.data[grad_above_thresh] *= alpha
                if DEBUG:
                    grad_pruned_cumsum += param.grad.data.abs().sum()

        return grad_cumsum, grad_pruned_cumsum

    def train_client(self, id, global_round_num, client_dataloader, filtered_dataset, global_model, idlg=False, idlg_prep_exp=False):
        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)
        local_model.train()

        self.optimizer = torch.optim.Adam(local_model.parameters(), lr=self.lr)
        psnr = []
        ssim = []

        for epoch in trange(self.num_local_epochs, desc="Local epochs"):
            for (index, (img, label)) in enumerate(client_dataloader):
                img, label = img.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()
                predict = local_model(img)
                loss = self.criterion(predict, label)
                loss.backward()

                self.cur_loss = loss.item()
                # apply gradient pruning optionally
                if self.alpha < 1 and self.prune_method == "prune":
                    org_grad, clipped_grad = self.prune_gradients(local_model, thres=self.thres, alpha=self.alpha)
                    if DEBUG:
                        grad_cumsum = 0
                        for param in local_model.parameters():
                            if param.grad is not None:
                                grad_cumsum += param.grad.data.abs().sum()
                        assert grad_cumsum == clipped_grad
                        print(f"Gradient cumsum: {org_grad}, Gradient cumsum after pruning: {clipped_grad}")
                elif self.prune_method == "clip":
                    self.clip_gradients(local_model, self.clip_value)
                    if DEBUG:
                        parm_grad = [p.grad for p in local_model.parameters() if p.grad is not None]
                        assert torch.nn.utils.get_total_norm(parm_grad, norm_type=2, foreach=True) - self.clip_value < 1e-6

                self.optimizer.step()

            if epoch == 0 and global_round_num == 0 and id == 1:
                # model needed for iDLG/InvGrad attacks
                ckpt_prefix = 'client_1_vit_' if config.model_name == 'vit' else 'client_1_'
                save_checkpoint(local_model, self.optimizer, self.cur_loss, epoch, self.alpha, self.thres, self.prune_method, prefix=ckpt_prefix)

            if epoch == 0 and (idlg or idlg_prep_exp is True) and global_round_num == 0 and id == 1:
                # Perform gradient inversion attack using iDLG:
                reconstructed_imgs = []
                ground_truth_imgs = []
                idlg_settings = []

                for idx in trange(len(filtered_dataset), desc="Reconstructing training images using iDLG"):
                    # Perform multiple reconstruction due to iDLG solving a non-convex optimization problem
                    results = {}
                    image, label = filtered_dataset[idx]
                    gt_data = image.to(self.device)
                    gt_data = gt_data.view(1, *gt_data.size())
                    gt_label = torch.tensor([label], dtype=torch.long).to(self.device)
                    gt_label = gt_label.view(1,)
                    idlg_settings_ = {
                        'orig_img': image,
                        'gt_data': gt_data,
                        'label': gt_label,
                    }

                    idlg_settings.append(idlg_settings_)

                    if idlg is False:
                        continue

                    for attempt in range(15):

                        idlg = ModifiediDLG(model=local_model, device=self.device, **idlg_settings_)
                        dummy_data, label_pred, history, losses, final_grad_diff, psnr_vals, ssim_vals = idlg.attack()
                        results[attempt] = (history[-1], final_grad_diff, psnr_vals[-1], ssim_vals[-1])

                    psnr_values = [result[2] for result in results.values() if not np.isnan(result[2])]
                    ssim_values = [result[3] for result in results.values() if not np.isnan(result[3])]

                    median_psnr = np.median(psnr_values) if psnr_values else float('nan')
                    median_ssim = np.median(ssim_values) if ssim_values else float('nan')

                    def combined_metric(result):
                        psnr_diff = abs(result[2] - median_psnr) if not np.isnan(result[2]) else float('inf')
                        ssim_diff = abs(result[3] - median_ssim) if not np.isnan(result[3]) else float('inf')
                        return psnr_diff + ssim_diff

                    best_attempt_key = min(results, key=lambda k: combined_metric(results[k]))
                    best_attempt = results[best_attempt_key]
                    best_reconstructed_image = best_attempt[0]
                    reconstructed_imgs.append(best_reconstructed_image)
                    ground_truth_imgs.append(image)

                    psnr.append(np.mean(psnr_values))
                    ssim.append(np.mean(ssim_values))

                if idlg_prep_exp:
                    save_idlg_exp_settings(idlg_settings)

                if idlg is False:
                    continue

                ground_truth_imgs_for_plotting = [prepare_tensor_for_plotting(img.squeeze(0)) for img in ground_truth_imgs]

                n = len(reconstructed_imgs)
                nrows = (n + 4) // 5
                plt.figure(figsize=(20, 4 * nrows))

                for i in range(n):
                    plt.subplot(nrows, 10, 2*i + 1)
                    plt.imshow(ground_truth_imgs_for_plotting[i])
                    plt.title(f"GT {i}")
                    plt.axis('off')

                    plt.subplot(nrows, 10, 2*i + 2)
                    plt.imshow(reconstructed_imgs[i])
                    plt.title(f"Recon {i}")
                    plt.axis('off')

                plt.tight_layout()

                if self.alpha is None:
                    save_path = os.path.join(os.getcwd(), "plots", "original")
                    os.makedirs(save_path, exist_ok=True)
                    save_path = os.path.join(save_path, f"client_{id}_iDLG.png")

                else:
                    save_path = os.path.join(os.getcwd(), "plots", f"alpha_{self.alpha}")
                    os.makedirs(save_path, exist_ok=True)
                    save_path = os.path.join(save_path, f"client_{id}_iDLG.png")

                plt.savefig(save_path, dpi=600)
                plt.close()

        if psnr and ssim:
            return local_model, np.mean(psnr), np.mean(ssim)
        else:
            return local_model, None, None

    def global_model_average(self, curr, next, scale):
        if curr is None:
            curr = next
            for key in curr:
                curr[key] = curr[key]*scale
        else:
            for key in curr:
                curr[key] = curr[key] + (next[key]*scale)
        return curr

    def test_accuracy(self, model, test_dataloader, device):
        model = model.to(device)
        model.eval()
        num_correct = 0
        total = 0
        with torch.no_grad():
            for (index, (img, label)) in enumerate(test_dataloader):
                img, label = img.to(device), label.to(device)
                predict = model(img)
                num_correct += torch.sum(torch.argmax(predict, dim=1) == label).item()
                total += img.shape[0]
        accuracy = num_correct / total
        return accuracy

    def train_accuracy(self, model, train_dataloader, device):
        model = model.to(device)
        model.eval()
        num_correct = 0
        total = 0
        with torch.no_grad():
            for (index, (img, label)) in enumerate(train_dataloader):
                img, label = img.to(device), label.to(device)
                predict = model(img)
                num_correct += torch.sum(torch.argmax(predict, dim=1) == label).item()
                total += img.shape[0]
        accuracy = num_correct / total
        return accuracy

    def federated_learning_experiment(self, idlg=False, idlg_prep_exp=False):
        round_train_accuracy = []
        round_test_accuracy = []
        ssim_vals = []
        psnr_vals = []

        global_model = self.global_model

        for round in trange(self.max_rounds, desc="Training rounds"):
            clients = np.random.choice(np.arange(5), self.num_clients_per_round, replace=False)
            # if client id 1 is not in the selected clients, add it since the idlg code only runs on client 1
            if round == 0 and 1 not in clients:
                clients[0] = 1

            global_model.eval()
            global_model = global_model.to(self.device)
            running_avg = None

            client_ssim = []
            client_psnr = []

            for index, client in enumerate(clients):
                local_model, psnr, ssim = self.train_client(client, round, self.clients_dataloader[client], self.filtered_train_dataset[client], global_model, idlg=idlg, idlg_prep_exp=idlg_prep_exp)
                running_avg = self.global_model_average(running_avg, local_model.state_dict(), 1/self.num_clients_per_round)

                if ssim is not None:
                    client_ssim.append(ssim)
                if psnr is not None:
                    client_psnr.append(psnr)

            global_model.load_state_dict(running_avg)
            test_accuracy_ = self.test_accuracy(global_model, self.test_dataloader, self.device)
            train_accuracy_ = self.train_accuracy(global_model, self.train_dataloader, self.device)
            round_train_accuracy.append(train_accuracy_)
            round_test_accuracy.append(test_accuracy_)
            if len(client_ssim) > 0:
                ssim_vals.append(np.mean(client_ssim))
            if len(client_psnr) > 0:
                psnr_vals.append(np.mean(client_psnr))

            if round == self.max_rounds - 1:
                save_checkpoint(self.global_model, self.optimizer, self.cur_loss, round+1, self.alpha, self.thres, self.prune_method)

        return round_train_accuracy, round_test_accuracy, psnr_vals, ssim_vals
