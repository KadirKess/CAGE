import torch
from cage.losses import mse_loss, ms_ssim_loss, ce_loss, cage_loss


def test_mse_loss_is_scalar():
    pred = torch.rand(4, 1, 28, 28)
    true = torch.rand(4, 1, 28, 28)
    assert mse_loss(pred, true).shape == torch.Size([])


def test_mse_loss_zero_on_identical_inputs():
    x = torch.rand(4, 1, 28, 28)
    assert mse_loss(x, x).item() == 0.0


def test_ce_loss_is_scalar():
    logits = torch.rand(4, 10)
    labels = torch.randint(0, 10, (4,))
    assert ce_loss(logits, labels).shape == torch.Size([])


def test_cage_loss_returns_three_scalars():
    pred = torch.rand(4, 1, 28, 28)
    true = torch.rand(4, 1, 28, 28)
    logits = torch.rand(4, 10)
    labels = torch.randint(0, 10, (4,))
    total, l_mse, l_ce = cage_loss(pred, true, logits, labels, lambda1=0.5, lambda2=0.5)
    for t in (total, l_mse, l_ce):
        assert t.shape == torch.Size([])


def test_cage_loss_is_weighted_sum():
    pred = torch.rand(4, 1, 28, 28)
    true = torch.rand(4, 1, 28, 28)
    logits = torch.rand(4, 10)
    labels = torch.randint(0, 10, (4,))
    total, l_mse, l_ce = cage_loss(pred, true, logits, labels, lambda1=0.7, lambda2=0.3)
    assert torch.isclose(total, 0.7 * l_mse + 0.3 * l_ce)


def test_ms_ssim_loss_is_scalar():
    pred = torch.rand(4, 3, 128, 128)
    true = torch.rand(4, 3, 128, 128)
    assert ms_ssim_loss(pred, true).shape == torch.Size([])


def test_ms_ssim_loss_zero_on_identical_inputs():
    x = torch.rand(4, 3, 128, 128)
    assert ms_ssim_loss(x, x).item() < 1e-5


def test_cage_loss_accepts_rec_loss_fn():
    pred = torch.rand(4, 3, 128, 128)
    true = torch.rand(4, 3, 128, 128)
    logits = torch.rand(4, 10)
    labels = torch.randint(0, 10, (4,))
    total, l_rec, l_ce = cage_loss(
        pred, true, logits, labels, lambda1=0.5, lambda2=0.5, rec_loss_fn=ms_ssim_loss
    )
    for t in (total, l_rec, l_ce):
        assert t.shape == torch.Size([])
    assert torch.isclose(total, 0.5 * l_rec + 0.5 * l_ce)


def test_fp16_losses_compute_in_fp32_and_preserve_gradients():
    pred = torch.rand(2, 3, 128, 128, dtype=torch.float16, requires_grad=True)
    true = torch.rand_like(pred)
    logits = torch.rand(2, 11, dtype=torch.float16, requires_grad=True)
    labels = torch.tensor([0, 1])
    with torch.autocast("cpu", dtype=torch.float16):
        losses = [mse_loss(pred, true), ms_ssim_loss(pred, true), ce_loss(logits, labels)]
    assert all(loss.dtype == torch.float32 and torch.isfinite(loss) for loss in losses)
    sum(losses).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
