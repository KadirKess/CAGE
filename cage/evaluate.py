"""Paired test evaluation of the study's denoisers: accuracy and SSIM per sigma."""

from typing import Mapping

import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure

from cage.data import get_dataloaders
from cage.models.cnn import WeakCNN
from cage.models.unet import UNet


class IndexedDataset(Dataset):
    """Attach stable sample indices so corruption can be regenerated exactly."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        return x, y, index


def paired_gaussian_noise(
    x: torch.Tensor, indices: torch.Tensor, sigma: float, evaluation_seed: int
) -> torch.Tensor:
    """Create a fixed corruption keyed by sample index and evaluation seed.

    Generating on CPU makes the same input reproducible across CPU, MPS, and
    CUDA, then the caller moves it to the selected device.
    """
    if sigma == 0:
        return x.clone()
    noisy = torch.empty_like(x, device="cpu")
    for row, sample_index in enumerate(indices.tolist()):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(evaluation_seed) * 1_000_003 + int(sample_index) * 9_176 + int(round(sigma * 1_000_000)))
        noise = torch.randn(x[row].shape, generator=generator, dtype=x.dtype)
        noisy[row] = (x[row].cpu() + noise * sigma).clamp(0.0, 1.0)
    return noisy


def evaluate_study_models(
    *,
    cnn_checkpoint: str,
    model_checkpoints: Mapping[str, str],
    sigmas: list[float],
    evaluation_seed: int,
    data_dir: str = "data",
    batch_size: int = 128,
    num_workers: int = 0,
    device_str: str = "cpu",
) -> dict:
    """Evaluate all denoisers against paired test corruptions in one pass.

    Every model receives identical noisy values for a sample/sigma pair, which
    makes downstream accuracy and SSIM differences meaningful and paired.
    """
    device = torch.device(device_str)
    loaders = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size, num_workers=num_workers,
    )
    test_loader = DataLoader(
        IndexedDataset(loaders["test"].dataset), batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    cnn = WeakCNN().to(device)
    cnn.load_state_dict(torch.load(cnn_checkpoint, map_location=device))
    cnn.freeze()
    models: dict[str, UNet] = {}
    for name, checkpoint in model_checkpoints.items():
        model = UNet().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        models[name] = model

    results: dict[str, dict] = {}
    with torch.no_grad():
        for sigma in sigmas:
            metrics = {
                name: {"correct": 0, "total": 0, "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(device)}
                for name in ("clean", "noisy", *models.keys())
            }
            for x, y, indices in test_loader:
                x = x.to(device)
                y = y.to(device).long().view(-1)
                noisy = paired_gaussian_noise(x.cpu(), indices, sigma, evaluation_seed).to(device)
                outputs: dict[str, torch.Tensor] = {"clean": x, "noisy": noisy}
                outputs.update({name: model(noisy) for name, model in models.items()})
                for name, output in outputs.items():
                    metrics[name]["correct"] += (cnn(output).argmax(dim=1) == y).sum().item()
                    metrics[name]["total"] += y.size(0)
                    metrics[name]["ssim"].update(output, x)
            results[str(sigma)] = {
                name: {
                    "accuracy": round(item["correct"] / item["total"], 6),
                    "ssim": round(item["ssim"].compute().item(), 6),
                }
                for name, item in metrics.items()
            }
    return results
