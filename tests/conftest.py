import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def batch():
    """A batch of 8 clean Imagenette-sized RGB images with 10-class labels."""
    x = torch.rand(8, 3, 128, 128)
    y = torch.randint(0, 10, (8,))
    return x, y


@pytest.fixture
def tiny_loaders():
    """16-sample synthetic Imagenette-shaped loaders for fast tests."""
    x = torch.rand(16, 3, 128, 128)
    y = torch.randint(0, 10, (16,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=8)
    return {"train": loader, "val": loader, "test": loader}


@pytest.fixture(autouse=True)
def patch_dataloaders(monkeypatch, tiny_loaders):
    """Patch get_dataloaders everywhere it's imported so tests never hit disk."""
    monkeypatch.setattr("cage.trainer.get_dataloaders", lambda **kw: tiny_loaders)
    monkeypatch.setattr("cage.evaluate.get_dataloaders", lambda **kw: tiny_loaders)
