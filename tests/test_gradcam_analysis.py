import pytest
import torch
from torch.utils.data import TensorDataset

from cage.gradcam_analysis import gradcam_metrics, clean_reference, entropy, iou, pearson, top_mask
from cage.models.cnn import WeakCNN
from cage.run import _gradcam_cases


def test_logit_contributions_sum_to_logit(batch):
    x, y = batch
    model = WeakCNN()
    model.freeze()
    with torch.no_grad():
        logits = model(x).gather(1, y[:, None])[:, 0]
        terms = model.logit_contributions(x, y)
    assert terms.shape == (8, 30, 30)
    assert torch.allclose(terms.flatten(1).sum(1) + model.classifier.bias[y], logits, atol=1e-4)


def test_top_mask_keeps_requested_fraction():
    maps = torch.rand(3, 32, 32)
    assert top_mask(maps, 0.2).flatten(1).sum(1).tolist() == [204.0] * 3


def test_identical_maps_correlate_and_overlap_fully():
    maps = torch.rand(2, 32, 32)
    assert torch.allclose(pearson(maps, maps), torch.ones(2))
    assert torch.allclose(iou(maps, maps), torch.ones(2))


def test_entropy_is_one_for_uniform_and_zero_for_a_point():
    point = torch.zeros(1, 32, 32)
    point[0, 3, 4] = 1.0
    assert entropy(torch.ones(1, 32, 32)).item() == pytest.approx(1.0)
    assert entropy(point).item() == pytest.approx(0.0, abs=1e-6)


def test_flat_maps_are_left_out():
    flat, varied = torch.zeros(1, 32, 32), torch.rand(1, 32, 32)
    assert pearson(flat, varied).isnan().all()
    assert iou(flat, varied).isnan().all()


def test_identity_denoiser_changes_nothing(batch, device):
    x, y = batch
    dataset = TensorDataset(x, y)
    cnn = WeakCNN()
    cnn.freeze()
    reference = clean_reference(cnn, dataset, 4, device)
    metrics = gradcam_metrics(cnn, torch.nn.Identity(), dataset, reference, 0.0, 1, 4, device)
    assert metrics["gradcam_corr"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["evidence_gain_total"] == pytest.approx(0.0, abs=1e-5)
    pasted = {metrics[f"accuracy_paste_{name}"] for name in ("none", "top", "shuffled", "all")}
    assert len(pasted) == 1


def test_gradcam_cases_probe_noise_free_models_on_clean_input():
    manifests = [
        {"config": {"id": "weakcnn", "role": "baseline"}},
        {"config": {"id": "mse_main", "role": "main", "train_sigmas": [0.3]}},
        {"config": {"id": "mse_sigma_000", "role": "sigma", "train_sigmas": [0.0]}},
        {"config": {"id": "mse_sigma_050", "role": "sigma", "train_sigmas": [0.5]}},
    ]
    assert _gradcam_cases(manifests) == {"mse_main": 0.3, "mse_sigma_000": 0.0}
