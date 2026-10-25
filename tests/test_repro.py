import torch

from cage.evaluate import paired_gaussian_noise
from cage.repro import seed_everything, write_manifest


def test_paired_noise_is_repeatable_and_indexed():
    x = torch.full((2, 1, 8, 8), 0.5)
    indices = torch.tensor([4, 9])
    first = paired_gaussian_noise(x, indices, sigma=0.3, evaluation_seed=7)
    second = paired_gaussian_noise(x, indices, sigma=0.3, evaluation_seed=7)
    changed = paired_gaussian_noise(x, indices, sigma=0.3, evaluation_seed=8)
    assert torch.equal(first, second)
    assert not torch.equal(first, changed)
    assert torch.equal(first[0], paired_gaussian_noise(x[0:1], torch.tensor([4]), 0.3, 7)[0])


def test_paired_noise_zero_sigma_is_clean_copy():
    x = torch.rand(2, 1, 8, 8)
    result = paired_gaussian_noise(x, torch.tensor([0, 1]), sigma=0.0, evaluation_seed=4)
    assert torch.equal(x, result)
    assert result.data_ptr() != x.data_ptr()


def test_manifest_records_provenance(tmp_path):
    state = seed_everything(3)
    manifest = write_manifest(tmp_path / "manifest.json", config={"id": "test"}, device="cpu", extra={"seed_state": state})
    assert manifest["config"]["id"] == "test"
    assert state["deterministic_algorithms_enabled"] is False
    assert manifest["packages"]["torch"] != "not-installed"
    assert (tmp_path / "manifest.json").exists()
