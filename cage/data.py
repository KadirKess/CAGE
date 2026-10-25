"""Imagenette (128x128 RGB) DataLoader construction and Gaussian noise utilities for CAGE."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DATASET_FILE = "imagenette_128.npz"
NUM_CLASSES = 10
CHANNELS = 3


class ImagenetteSplit(Dataset):
    """One split held in memory as uint8 CHW; items are float in [0, 1]."""

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2)))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index].float().div_(255.0), self.labels[index]

    def __getitems__(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        # DataLoader calls this with a whole batch of indices. One gather plus
        # one conversion is about 10x faster than collating 64 single items
        index = torch.as_tensor(indices)
        return self.images[index].float().div_(255.0), self.labels[index]


def _batch_collate(batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return batch


def load_splits(data_dir: str = "data") -> dict[str, ImagenetteSplit]:
    path = Path(data_dir) / DATASET_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; build it with: python scripts/prepare_imagenette.py --output {path}"
        )
    with np.load(path) as archive:
        return {
            split: ImagenetteSplit(archive[f"{split}_images"], archive[f"{split}_labels"])
            for split in ("train", "val", "test")
        }


def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 128,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Build loaders over the prepared npz from ``scripts/prepare_imagenette.py``.

    The whole dataset sits in memory, so ``num_workers=0`` is usually enough;
    workers, when enabled, persist across epochs.
    """
    splits = load_splits(data_dir)
    loader_options: dict = {"num_workers": num_workers, "collate_fn": _batch_collate}
    if num_workers > 0:
        loader_options["persistent_workers"] = True
    return {
        "train": DataLoader(splits["train"], batch_size=batch_size, shuffle=True, **loader_options),
        "val": DataLoader(splits["val"], batch_size=batch_size, shuffle=False, **loader_options),
        "test": DataLoader(splits["test"], batch_size=batch_size, shuffle=False, **loader_options),
    }


def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    # Independent noise per pixel and per channel
    return (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)
