import torch
import torch.nn as nn
from cage.models.cnn import WeakCNN
from cage.models.unet import UNet


def test_cnn_output_shape(batch):
    x, _ = batch
    logits = WeakCNN()(x)
    assert logits.shape == (8, 10)


def test_cnn_has_conv1_and_conv2():
    model = WeakCNN()
    assert hasattr(model, "conv1")
    assert hasattr(model, "conv2")


def test_cnn_freeze_sets_no_grad():
    model = WeakCNN()
    model.freeze()
    for param in model.parameters():
        assert not param.requires_grad


def test_cnn_freeze_sets_eval_mode():
    model = WeakCNN()
    model.freeze()
    assert not model.training


def test_unet_output_shape(batch):
    x, _ = batch
    assert UNet()(x).shape == (8, 3, 128, 128)


def test_unet_output_in_unit_range(batch):
    x, _ = batch
    out = UNet()(x)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_unet_gradients_flow_through_frozen_cnn():
    """Gradients from the CNN CE loss must produce non-zero gradients in U-Net."""
    unet = UNet()
    cnn = WeakCNN()
    cnn.freeze()

    x = torch.rand(4, 3, 128, 128)
    y = torch.randint(0, 10, (4,))

    x_denoised = unet(x)
    logits = cnn(x_denoised)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in unet.parameters()
    )
    assert has_grad
