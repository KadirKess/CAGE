import json
import os
from pathlib import Path

import pytest
import torch

from cage.models.cnn import WeakCNN
from cage.trainer import Trainer

BASE_CONFIG = {
    "epochs": 1,
    "lr": 1e-3,
    "noise_sigmas": [0.3],
    "lambda1": 0.5,
    "lambda2": 0.5,
    "device": "cpu",
    "batch_size": 32,
}


def _save_dummy_cnn(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(WeakCNN().state_dict(), path)


def test_cnn_mode_prints_status_lines_without_progress_bars(tmp_path, capsys):
    config = {
        **BASE_CONFIG,
        "mode": "cnn",
        "run_name": "t_cnn",
        "metrics_dir": str(tmp_path / "metrics"),
        "checkpoints_dir": str(tmp_path / "checkpoints"),
        "plots_dir": str(tmp_path / "plots"),
    }
    logs = Trainer(config).train()
    assert len(logs) == 1
    assert "train_loss" in logs[0] and "val_loss" in logs[0]
    output = capsys.readouterr()
    lines = output.out.splitlines()
    assert lines[0].startswith("t_cnn: device=cpu, precision=fp32")
    assert any(line.startswith("t_cnn: epoch 1/1 train=") and "patience=0/20" in line for line in lines)
    assert lines[-1].startswith("t_cnn: finished at epoch 1")
    assert "\r" not in output.out + output.err


def test_cnn_mode_saves_checkpoint(tmp_path):
    config = {
        **BASE_CONFIG,
        "mode": "cnn",
        "run_name": "t_cnn_ckpt",
        "metrics_dir": str(tmp_path / "metrics"),
        "checkpoints_dir": str(tmp_path / "checkpoints"),
        "plots_dir": str(tmp_path / "plots"),
    }
    Trainer(config).train()
    assert os.path.exists(str(tmp_path / "checkpoints" / "t_cnn_ckpt.pt"))


def test_unet_mse_mode_logs_accuracy_per_sigma(tmp_path):
    cnn_ckpt = str(tmp_path / "checkpoints" / "cnn.pt")
    _save_dummy_cnn(cnn_ckpt)
    config = {
        **BASE_CONFIG,
        "mode": "unet_mse",
        "run_name": "t_unet_mse",
        "cnn_checkpoint": cnn_ckpt,
        "metrics_dir": str(tmp_path / "metrics"),
        "checkpoints_dir": str(tmp_path / "checkpoints"),
        "plots_dir": str(tmp_path / "plots"),
    }
    logs = Trainer(config).train()
    assert "cnn_accuracy_per_sigma" in logs[0]
    assert "0.3" in logs[0]["cnn_accuracy_per_sigma"]


def test_cage_mode_logs_mode_name(tmp_path):
    cnn_ckpt = str(tmp_path / "checkpoints" / "cnn.pt")
    _save_dummy_cnn(cnn_ckpt)
    config = {
        **BASE_CONFIG,
        "mode": "cage",
        "run_name": "t_cage",
        "cnn_checkpoint": cnn_ckpt,
        "metrics_dir": str(tmp_path / "metrics"),
        "checkpoints_dir": str(tmp_path / "checkpoints"),
        "plots_dir": str(tmp_path / "plots"),
    }
    logs = Trainer(config).train()
    assert logs[0]["mode"] == "cage"
    assert "cnn_accuracy_per_sigma" in logs[0]


def test_checkpoint_saves_tiny_improvements_independently_of_patience(tmp_path, monkeypatch):
    config = {
        **BASE_CONFIG,
        "mode": "cnn",
        "run_name": "t_early_stop_min_delta",
        "epochs": 10,
        "early_stopping_patience": 2,
        "early_stopping_min_delta": 1e-4,
        "metrics_dir": str(tmp_path / "metrics"),
        "checkpoints_dir": str(tmp_path / "checkpoints"),
        "plots_dir": str(tmp_path / "plots"),
    }
    trainer = Trainer(config)

    val_losses = iter([1.0, 0.99995, 0.99994, 0.99993])
    monkeypatch.setattr(trainer, "_train_epoch", lambda: (1.0, None, None))
    validation_epoch = 0

    def validate():
        nonlocal validation_epoch
        validation_epoch += 1
        with torch.no_grad():
            trainer.cnn.conv1.weight.fill_(validation_epoch)
        return next(val_losses), None, None, {}

    monkeypatch.setattr(trainer, "_val_epoch", validate)

    logs = trainer.train()

    assert len(logs) == 3
    assert logs[-1]["best_val_loss"] == 0.99994
    assert logs[-1]["best_epoch"] == 3
    saved = torch.load(tmp_path / "checkpoints" / "t_early_stop_min_delta.pt", weights_only=True)
    assert torch.equal(saved["conv1.weight"], torch.full_like(saved["conv1.weight"], 3))


def test_unet_mse_msssim_mode_runs(tmp_path):
    cnn_ckpt = str(tmp_path / "checkpoints" / "cnn.pt")
    _save_dummy_cnn(cnn_ckpt)
    config = {
        **BASE_CONFIG,
        "mode": "unet_mse",
        "rec_loss": "msssim",
        "run_name": "t_unet_msssim",
        "cnn_checkpoint": cnn_ckpt,
        "metrics_dir": str(tmp_path / "metrics"),
        "checkpoints_dir": str(tmp_path / "checkpoints"),
        "plots_dir": str(tmp_path / "plots"),
    }
    logs = Trainer(config).train()
    assert len(logs) == 1
    assert "train_loss" in logs[0]
    assert logs[0]["rec_loss"] == "msssim"


def test_cage_msssim_mode_logs_rec_loss_key(tmp_path):
    cnn_ckpt = str(tmp_path / "checkpoints" / "cnn.pt")
    _save_dummy_cnn(cnn_ckpt)
    config = {
        **BASE_CONFIG,
        "mode": "cage",
        "rec_loss": "msssim",
        "run_name": "t_cage_msssim",
        "cnn_checkpoint": cnn_ckpt,
        "metrics_dir": str(tmp_path / "metrics"),
        "checkpoints_dir": str(tmp_path / "checkpoints"),
        "plots_dir": str(tmp_path / "plots"),
    }
    logs = Trainer(config).train()
    assert logs[0]["rec_loss"] == "msssim"
    assert "train_rec_loss" in logs[0]


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("mode,rec_loss", [
    ("cnn", "mse"), ("unet_mse", "mse"), ("unet_mse", "msssim"),
    ("cage", "mse"), ("cage", "msssim"),
])
def test_fp16_training_keeps_finite_weights_and_frozen_critic(tmp_path, device, mode, rec_loss):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable in this process")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable in this process")
    checkpoint = tmp_path / "critic.pt"
    _save_dummy_cnn(str(checkpoint))
    trainer = Trainer({
        **BASE_CONFIG, "mode": mode, "rec_loss": rec_loss, "device": device,
        "precision": "fp16", "run_name": "amp_smoke", "cnn_checkpoint": str(checkpoint),
        "metrics_dir": str(tmp_path / "metrics"),
        "checkpoints_dir": str(tmp_path / "checkpoints"), "plots_dir": str(tmp_path / "plots"),
    })
    model = trainer.cnn if mode == "cnn" else trainer.unet
    initial = {name: param.detach().clone() for name, param in model.named_parameters()}
    critic = {name: param.detach().clone() for name, param in trainer.cnn.named_parameters()}
    output_dtypes = []
    handle = model.register_forward_hook(lambda module, inputs, output: output_dtypes.append(output.dtype))
    logs = trainer.train()
    handle.remove()
    assert output_dtypes and set(output_dtypes) == {torch.float16}
    assert trainer.scaler.is_enabled()
    assert logs[0]["precision"] == "fp16"
    assert all(param.dtype == torch.float32 and torch.isfinite(param).all() for param in model.parameters())
    assert any(not torch.equal(param, initial[name]) for name, param in model.named_parameters())
    if mode != "cnn":
        assert all(not param.requires_grad and param.grad is None for param in trainer.cnn.parameters())
        assert all(torch.equal(param, critic[name]) for name, param in trainer.cnn.named_parameters())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["cnn", "cage"])
def test_bf16_training_runs_without_gradient_scaling(tmp_path, device, mode):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable in this process")
    checkpoint = tmp_path / "critic.pt"
    _save_dummy_cnn(str(checkpoint))
    trainer = Trainer({
        **BASE_CONFIG, "mode": mode, "device": device, "precision": "bf16", "tf32": True,
        "run_name": "bf16_smoke", "cnn_checkpoint": str(checkpoint),
        "metrics_dir": str(tmp_path / "metrics"),
        "checkpoints_dir": str(tmp_path / "checkpoints"), "plots_dir": str(tmp_path / "plots"),
    })
    model = trainer.cnn if mode == "cnn" else trainer.unet
    initial = {name: param.detach().clone() for name, param in model.named_parameters()}
    output_dtypes = []
    handle = model.register_forward_hook(lambda module, inputs, output: output_dtypes.append(output.dtype))
    logs = trainer.train()
    handle.remove()
    assert output_dtypes and set(output_dtypes) == {torch.bfloat16}
    assert not trainer.scaler.is_enabled()
    assert logs[0]["precision"] == "bf16" and logs[0]["tf32"] is True
    assert all(param.dtype == torch.float32 and torch.isfinite(param).all() for param in model.parameters())
    assert any(not torch.equal(param, initial[name]) for name, param in model.named_parameters())


def test_bf16_nonfinite_gradients_abort_instead_of_skipping():
    trainer = Trainer({**BASE_CONFIG, "mode": "cnn", "run_name": "bf16_inf", "precision": "bf16"})
    trainer.cnn.conv1.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    with pytest.raises(FloatingPointError, match="non-finite gradients"):
        trainer._train_epoch()


def test_nonfinite_validation_preserves_last_good_checkpoint(tmp_path, monkeypatch):
    trainer = Trainer({
        **BASE_CONFIG, "epochs": 2, "mode": "cnn", "run_name": "nonfinite",
        "metrics_dir": str(tmp_path / "metrics"),
        "checkpoints_dir": str(tmp_path / "checkpoints"), "plots_dir": str(tmp_path / "plots"),
    })
    losses = iter([1.0, float("nan")])
    monkeypatch.setattr(trainer, "_train_epoch", lambda: (1.0, None, None))
    monkeypatch.setattr(trainer, "_val_epoch", lambda: (next(losses), None, None, {}))
    with pytest.raises(FloatingPointError, match="non-finite epoch loss"):
        trainer.train()
    assert (tmp_path / "checkpoints" / "nonfinite.pt").exists()
    logs = json.loads((tmp_path / "metrics" / "nonfinite.json").read_text())
    assert len(logs) == 1


def test_fp16_overflow_skips_updates_and_reduces_scale(tmp_path):
    trainer = Trainer({**BASE_CONFIG, "mode": "cnn", "run_name": "overflow", "precision": "fp16"})
    original = {name: param.detach().clone() for name, param in trainer.cnn.named_parameters()}
    trainer.cnn.conv1.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    with pytest.raises(FloatingPointError, match="every optimizer step overflowed"):
        trainer._train_epoch()
    assert trainer._skipped_steps == len(trainer.loaders["train"])
    assert trainer.scaler.get_scale() < 1024
    assert all(torch.equal(param, original[name]) for name, param in trainer.cnn.named_parameters())


def test_plateau_scheduler_reduces_lr_after_patience_and_logs_it(tmp_path, monkeypatch):
    trainer = Trainer({
        **BASE_CONFIG, "epochs": 12, "mode": "cnn", "run_name": "plateau",
        "metrics_dir": str(tmp_path / "metrics"),
        "checkpoints_dir": str(tmp_path / "checkpoints"), "plots_dir": str(tmp_path / "plots"),
    })
    monkeypatch.setattr(trainer, "_train_epoch", lambda: (1.0, None, None))
    monkeypatch.setattr(trainer, "_val_epoch", lambda: (1.0, None, None, {}))
    logs = trainer.train()
    assert all(row["learning_rate"] == BASE_CONFIG["lr"] for row in logs[:11])
    assert logs[11]["learning_rate"] == BASE_CONFIG["lr"] * 0.5
    assert logs[-1]["best_epoch"] == 1


def test_compiled_fp16_cage_preserves_gradients_and_checkpoint_keys(tmp_path):
    checkpoint = tmp_path / "critic.pt"
    _save_dummy_cnn(str(checkpoint))
    trainer = Trainer({
        **BASE_CONFIG, "mode": "cage", "run_name": "compiled", "precision": "fp16",
        "compile": True, "compile_backend": "aot_eager", "cnn_checkpoint": str(checkpoint),
        "metrics_dir": str(tmp_path / "metrics"),
        "checkpoints_dir": str(tmp_path / "checkpoints"), "plots_dir": str(tmp_path / "plots"),
    })
    original = trainer.unet.out[0].weight.detach().clone()
    critic = {name: param.detach().clone() for name, param in trainer.cnn.named_parameters()}
    logs = trainer.train()
    assert not torch.equal(trainer.unet.out[0].weight, original)
    assert all(torch.equal(param, critic[name]) for name, param in trainer.cnn.named_parameters())
    assert logs[0]["compile"] and logs[0]["epoch_seconds"] > 0
    saved = torch.load(tmp_path / "checkpoints" / "compiled.pt", weights_only=True)
    from cage.models.unet import UNet

    UNet().load_state_dict(saved)
    assert all(not key.startswith("_orig_mod.") for key in saved)
