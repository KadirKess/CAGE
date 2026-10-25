import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim as _ms_ssim_fn

_mse_fn = nn.MSELoss()
_ce_fn = nn.CrossEntropyLoss()


def mse_loss(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=x_pred.device.type, enabled=False):
        return _mse_fn(x_pred.float(), x_true.float())


def ms_ssim_loss(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    # The default window needs inputs larger than 160 pixels.
    with torch.autocast(device_type=x_pred.device.type, enabled=False):
        return 1.0 - _ms_ssim_fn(x_pred.float(), x_true.float(), data_range=1.0, size_average=True, win_size=7)


def ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=logits.device.type, enabled=False):
        return _ce_fn(logits.float(), labels)


def cage_loss(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    lambda1: float,
    lambda2: float,
    rec_loss_fn=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rec_loss_fn is None:
        rec_loss_fn = mse_loss
    l_rec = rec_loss_fn(x_pred, x_true)
    l_ce = ce_loss(logits, labels)
    return lambda1 * l_rec + lambda2 * l_ce, l_rec, l_ce
