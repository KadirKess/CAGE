"""GradCAM heatmap computation for WeakCNN (cage/models/cnn.py)."""

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def compute_gradcam(
    model,
    input_tensor: torch.Tensor,
    target_classes: list[int] | None = None,
) -> np.ndarray:
    """Compute GradCAM heatmaps for a batch of images using WeakCNN.conv2.

    Args:
        model: WeakCNN instance (frozen or not — state is preserved after call)
        input_tensor: (N, 3, 128, 128) float tensor in [0, 1]
        target_classes: list of N class indices; None uses predicted class

    Returns:
        heatmaps: (N, 128, 128) float32 numpy array, values in [0, 1]
    """
    targets = (
        [ClassifierOutputTarget(c) for c in target_classes]
        if target_classes is not None
        else None
    )

    # pytorch-grad-cam needs grad-enabled parameters, so save and restore the state
    saved_requires_grad = {n: p.requires_grad for n, p in model.named_parameters()}
    was_training = model.training
    for p in model.parameters():
        p.requires_grad_(True)

    try:
        with GradCAM(model=model, target_layers=[model.conv2]) as cam:
            heatmaps = cam(input_tensor=input_tensor, targets=targets)
    finally:
        for n, p in model.named_parameters():
            p.requires_grad_(saved_requires_grad[n])
        if not was_training:
            model.eval()

    return heatmaps  # (N, 128, 128)
