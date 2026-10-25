"""JSON epoch logging and matplotlib plot helpers for CAGE experiment artifacts."""

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def as_image(image: torch.Tensor) -> np.ndarray:
    """CHW tensor in [0, 1] -> HWC array for imshow."""
    return image.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)


def signed_difference(diff: torch.Tensor) -> np.ndarray:
    """CHW difference -> HW signed map, averaged over channels.

    Positive means the output is brighter than the clean image there. Changes
    of opposite sign in different channels partly cancel.
    """
    return diff.detach().cpu().mean(dim=0).numpy()


def append_epoch_log(
    run_name: str,
    entry: dict[str, Any],
    metrics_dir: str = "outputs/metrics",
) -> None:
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(metrics_dir, f"{run_name}.json")
    logs: list = []
    if os.path.exists(path):
        with open(path) as f:
            logs = json.load(f)
    logs.append(entry)
    with open(path, "w") as f:
        json.dump(logs, f, indent=2)


def save_loss_curves(
    run_name: str,
    logs: list[dict],
    plots_dir: str = "outputs/plots",
) -> None:
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    epochs = [e["epoch"] for e in logs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [e["train_loss"] for e in logs], label="Train")
    ax.plot(epochs, [e["val_loss"] for e in logs], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curves — {run_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{run_name}_loss_curves.png"), dpi=150)
    plt.close(fig)


def save_accuracy_curves(
    run_name: str,
    logs: list[dict],
    plots_dir: str = "outputs/plots",
) -> None:
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    acc_logs = [e for e in logs if "cnn_accuracy_per_sigma" in e]
    if not acc_logs:
        return
    epochs = [e["epoch"] for e in acc_logs]
    sigmas = list(acc_logs[0]["cnn_accuracy_per_sigma"].keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    for sigma in sigmas:
        ax.plot(
            epochs,
            [e["cnn_accuracy_per_sigma"][sigma] for e in acc_logs],
            label=f"σ={sigma}",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CNN Accuracy")
    ax.set_title(f"CNN Accuracy per σ — {run_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{run_name}_accuracy_curves.png"), dpi=150)
    plt.close(fig)
