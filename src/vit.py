import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification


class ViTWrapper(nn.Module):
    """
    Wraps a HuggingFace ViT as a plain nn.Module that accepts
    (B, 3, H, W) float tensors in CIFAR-10 normalized space and returns logits (B, num_classes).

    The training pipeline applies CIFAR-10 normalization (mean/std from dataset.py).
    This wrapper converts those inputs to the normalization the ViT processor expects,
    then resizes to 224×224 using differentiable bilinear interpolation so that
    gradient-inversion attacks work without any changes to the rest of the pipeline.
    """

    def __init__(self, model_name="nielsr/vit-base-patch16-224-in21k-finetuned-cifar10"):
        super().__init__()
        self.vit = AutoModelForImageClassification.from_pretrained(model_name, attn_implementation="eager")
        self.vit.requires_grad_(True)

        processor = AutoImageProcessor.from_pretrained(model_name)
        self.target_size = processor.size.get("height", 224)

        # Normalization the ViT processor expects — read directly from processor config
        # so this stays correct if the model is swapped.
        # register_buffer so tensors move to GPU automatically with .to(device)
        self.register_buffer("vit_mean",   torch.tensor(processor.image_mean).view(1, 3, 1, 1))
        self.register_buffer("vit_std",    torch.tensor(processor.image_std).view(1, 3, 1, 1))
        # CIFAR-10 normalization applied by dataset.py
        self.register_buffer("cifar_mean", torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer("cifar_std",  torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def forward(self, x):
        # Undo CIFAR-10 normalization → [0, 1]
        x = x * self.cifar_std + self.cifar_mean
        # Resize 32×32 → 224×224 (differentiable, preserves gradients for attacks)
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                              mode="bilinear", align_corners=False)
        # Apply ViT processor normalization
        x = (x - self.vit_mean) / self.vit_std
        return self.vit(pixel_values=x).logits
