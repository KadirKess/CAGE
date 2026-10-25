import torch

from cage.evaluate import evaluate_study_models, paired_gaussian_noise
from cage.models.cnn import WeakCNN
from cage.models.unet import UNet


def _checkpoints(tmp_path):
    cnn = tmp_path / "cnn.pt"
    torch.save(WeakCNN().state_dict(), cnn)
    models = {}
    for name in ("mse_control", "mse_main"):
        path = tmp_path / f"{name}.pt"
        torch.save(UNet().state_dict(), path)
        models[name] = str(path)
    return str(cnn), models


def test_study_evaluation_reports_every_condition_per_sigma(tmp_path):
    cnn, models = _checkpoints(tmp_path)
    results = evaluate_study_models(
        cnn_checkpoint=cnn, model_checkpoints=models, sigmas=[0.0, 0.3], evaluation_seed=1, batch_size=8,
    )
    assert set(results) == {"0.0", "0.3"}
    for sigma in results.values():
        assert set(sigma) == {"clean", "noisy", "mse_control", "mse_main"}
        assert all(0.0 <= item["accuracy"] <= 1.0 and "ssim" in item for item in sigma.values())
    # At sigma 0 the noisy input is the clean image
    assert results["0.0"]["noisy"] == results["0.0"]["clean"]


def test_study_evaluation_is_repeatable_for_one_evaluation_seed(tmp_path):
    cnn, models = _checkpoints(tmp_path)
    kwargs = dict(cnn_checkpoint=cnn, model_checkpoints=models, sigmas=[0.3], batch_size=8)
    first = evaluate_study_models(evaluation_seed=5, **kwargs)
    assert evaluate_study_models(evaluation_seed=5, **kwargs) == first


def test_paired_noise_is_independent_per_channel():
    noisy = paired_gaussian_noise(torch.full((1, 3, 16, 16), 0.5), torch.tensor([0]), 0.1, 1)
    assert not torch.equal(noisy[0, 0], noisy[0, 1])
