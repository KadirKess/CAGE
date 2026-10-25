import torch
import numpy as np
from cage.models.cnn import WeakCNN
from cage.gradcam import compute_gradcam


def test_gradcam_output_shape():
    model = WeakCNN()
    x = torch.rand(4, 3, 128, 128)
    heatmaps = compute_gradcam(model, x)
    assert heatmaps.shape == (4, 128, 128)


def test_gradcam_values_in_unit_range():
    model = WeakCNN()
    x = torch.rand(4, 3, 128, 128)
    heatmaps = compute_gradcam(model, x)
    assert heatmaps.min() >= 0.0
    assert heatmaps.max() <= 1.0


def test_gradcam_with_explicit_target_classes():
    model = WeakCNN()
    x = torch.rand(3, 3, 128, 128)
    heatmaps = compute_gradcam(model, x, target_classes=[0, 5, 9])
    assert heatmaps.shape == (3, 128, 128)


def test_gradcam_works_on_frozen_model():
    model = WeakCNN()
    model.freeze()
    x = torch.rand(2, 3, 128, 128)
    heatmaps = compute_gradcam(model, x)
    assert heatmaps.shape == (2, 128, 128)
    # Model must remain frozen after call
    for param in model.parameters():
        assert not param.requires_grad
