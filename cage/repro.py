"""Reproducibility and provenance helpers for CAGE experiment runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_PACKAGES = ("torch", "torchvision", "numpy", "pillow", "torchmetrics", "pytorch-msssim")


def seed_everything(seed: int) -> dict[str, bool]:
    """Seed local PRNGs without forcing slow deterministic kernels.

    MPS is intentionally allowed to use its normal asynchronous, optimized
    kernels.  The seed and environment are still recorded in every manifest;
    exact bitwise replay is not promised across MPS/CUDA/CPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    return {
        "deterministic_algorithms_enabled": False,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }


def configure_tf32(enabled: bool) -> None:
    """Set CUDA TF32 for convolutions and matmuls explicitly.

    cuDNN allows TF32 convolutions by default, so "fp32" on an Ampere or newer
    GPU is not IEEE FP32 unless this is called with ``enabled=False``. The
    setting has no effect on CPU or MPS.
    """
    mode = "tf32" if enabled else "ieee"
    # torch >= 2.9 refuses to mix the fp32_precision API with allow_tf32
    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        torch.backends.cudnn.conv.fp32_precision = mode
        torch.backends.cuda.matmul.fp32_precision = mode
    else:
        torch.backends.cudnn.allow_tf32 = enabled
        torch.backends.cuda.matmul.allow_tf32 = enabled


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(
    path: str | Path,
    *,
    config: dict,
    device: str,
    checkpoint: str | Path | None = None,
    extra: dict | None = None,
) -> dict:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "device": device,
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
    }
    if checkpoint is not None and Path(checkpoint).exists():
        manifest["checkpoint"] = str(checkpoint)
        manifest["checkpoint_sha256"] = sha256_file(checkpoint)
    if extra:
        manifest.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
