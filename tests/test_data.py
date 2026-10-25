import numpy as np
import pytest
import torch

from cage.data import DATASET_FILE, add_gaussian_noise, get_dataloaders


@pytest.fixture
def prepared(tmp_path):
    """A tiny npz in the layout scripts/prepare_imagenette.py writes."""
    rng = np.random.default_rng(0)
    arrays = {}
    for split, n in (("train", 20), ("val", 6), ("test", 6)):
        arrays[f"{split}_images"] = rng.integers(0, 256, (n, 128, 128, 3), dtype=np.uint8)
        arrays[f"{split}_labels"] = rng.integers(0, 10, n).astype(np.int64)
    np.savez_compressed(tmp_path / DATASET_FILE, **arrays)
    return tmp_path, arrays


def test_loaders_cover_three_splits(prepared):
    data_dir, arrays = prepared
    loaders = get_dataloaders(data_dir=str(data_dir), batch_size=8)
    assert set(loaders) == {"train", "val", "test"}
    assert len(loaders["train"].dataset) == 20


def test_batches_are_rgb_chw_floats_in_unit_range(prepared):
    data_dir, arrays = prepared
    x, y = next(iter(get_dataloaders(data_dir=str(data_dir), batch_size=8)["val"]))
    assert x.shape == (6, 3, 128, 128)
    assert x.dtype == torch.float32 and y.dtype == torch.int64
    assert 0.0 <= x.min() and x.max() <= 1.0
    # HWC uint8 on disk becomes CHW in [0, 1] without reordering pixels
    expected = torch.from_numpy(arrays["val_images"][0]).permute(2, 0, 1).float() / 255
    assert torch.equal(x[0], expected)
    assert torch.equal(y, torch.from_numpy(arrays["val_labels"][: len(y)]))


def test_batched_and_single_item_access_agree(prepared):
    data_dir, _ = prepared
    dataset = get_dataloaders(data_dir=str(data_dir), batch_size=8)["train"].dataset
    images, labels = dataset.__getitems__([3, 7])
    assert torch.equal(images[1], dataset[7][0]) and labels[1] == dataset[7][1]


def test_missing_dataset_points_to_the_prepare_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_imagenette.py"):
        get_dataloaders(data_dir=str(tmp_path))


def test_workers_are_persistent_when_enabled(prepared):
    data_dir, _ = prepared
    loaders = get_dataloaders(data_dir=str(data_dir), batch_size=8, num_workers=1)
    assert loaders["train"].persistent_workers is True


def test_add_gaussian_noise_preserves_shape_and_clips():
    x = torch.rand(4, 3, 128, 128)
    noisy = add_gaussian_noise(x, sigma=5.0)
    assert noisy.shape == x.shape
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0


def test_add_gaussian_noise_changes_values():
    x = torch.rand(4, 3, 128, 128)
    assert not torch.allclose(x, add_gaussian_noise(x, sigma=0.3))
