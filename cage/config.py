"""Versioned, validated configuration for reproducible CAGE runs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

REQUIRED_FIELDS = {
    "id", "study", "mode", "rec_loss", "train_sigmas", "eval_sigmas", "seeds",
    "epochs", "lr", "batch_size", "data_dir", "num_workers", "device",
    "early_stopping_patience", "early_stopping_min_delta",
}
VALID_MODES = {"cnn", "unet_mse", "cage"}
VALID_REC_LOSSES = {"mse", "msssim"}


def load_config(path: str | Path) -> dict:
    """Load one JSON config and reject incomplete/ambiguous experiment definitions."""
    path = Path(path)
    config = json.loads(path.read_text())
    validate_config(config, source=str(path))
    return config


def load_configs(path: str | Path) -> list[dict]:
    """Load either one runnable config or the v0.1 study matrix."""
    raw = json.loads(Path(path).read_text())
    if "shared" not in raw:
        validate_config(raw, source=str(path))
        return [raw]
    return expand_study_config(raw, source=str(path))


def expand_study_config(study: dict, *, source: str = "study") -> list[dict]:
    """Expand the intentionally compact v0.1 matrix into runnable configurations."""
    shared = study.get("shared", {})
    seeds = study.get("seeds")
    eval_sigmas = study.get("eval_sigmas")
    if not isinstance(seeds, list) or not isinstance(eval_sigmas, list):
        raise ValueError(f"{source}: study requires seeds and eval_sigmas")
    configs: list[dict] = []

    def add(config_id: str, **overrides: object) -> None:
        config = {**shared, **overrides}
        config.update({"id": config_id, "study": study["study"], "seeds": seeds, "eval_sigmas": eval_sigmas})
        validate_config(config, source=f"{source}:{config_id}")
        configs.append(config)

    add("weakcnn", mode="cnn", rec_loss="mse", train_sigmas=[0.3], role="baseline")
    for fidelity in ("mse", "msssim"):
        rec_mode = "unet_mse"
        add(f"{fidelity}_control", mode=rec_mode, rec_loss=fidelity, train_sigmas=[0.3], role="control", fidelity=fidelity)
        for name, lambda1, lambda2 in (("lambda_000_100", 0.0, 1.0), ("lambda_025_075", 0.25, 0.75), ("main", 0.5, 0.5), ("lambda_075_025", 0.75, 0.25)):
            add(f"{fidelity}_{name}", mode="cage", rec_loss=fidelity, train_sigmas=[0.3], lambda1=lambda1, lambda2=lambda2, role="main" if name == "main" else "lambda", fidelity=fidelity)
        for name, sigmas in (("sigma_000", [0.0]), ("sigma_010", [0.1]), ("sigma_050", [0.5]), ("sigma_010_030", [0.1, 0.3]), ("sigma_010_030_050", [0.1, 0.3, 0.5])):
            add(f"{fidelity}_{name}", mode="cage", rec_loss=fidelity, train_sigmas=sigmas, lambda1=0.5, lambda2=0.5, role="sigma", fidelity=fidelity)
    return configs


def validate_config(config: dict, *, source: str = "config") -> None:
    missing = REQUIRED_FIELDS - config.keys()
    if missing:
        raise ValueError(f"{source}: missing required fields: {', '.join(sorted(missing))}")
    if config["mode"] not in VALID_MODES:
        raise ValueError(f"{source}: unsupported mode {config['mode']!r}")
    if config["rec_loss"] not in VALID_REC_LOSSES:
        raise ValueError(f"{source}: unsupported rec_loss {config['rec_loss']!r}")
    if config.get("precision", "fp32") not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"{source}: precision must be fp32, fp16 or bf16")
    if not isinstance(config.get("tf32", False), bool):
        raise ValueError(f"{source}: tf32 must be a boolean")
    if not isinstance(config.get("compile", False), bool):
        raise ValueError(f"{source}: compile must be a boolean")
    if config.get("compile_backend", "inductor") not in {"inductor", "aot_eager", "eager"}:
        raise ValueError(f"{source}: unsupported compile_backend")
    if config.get("max_grad_norm", 1.0) <= 0:
        raise ValueError(f"{source}: max_grad_norm must be positive")
    if config.get("amp_init_scale", 1024.0) <= 0:
        raise ValueError(f"{source}: amp_init_scale must be positive")
    if config["early_stopping_min_delta"] < 0:
        raise ValueError(f"{source}: early_stopping_min_delta must be non-negative")
    if config.get("lr_scheduler_patience", 10) < 0 or not 0 < config.get("lr_scheduler_factor", 0.5) < 1:
        raise ValueError(f"{source}: invalid plateau scheduler patience or factor")
    if not config["id"] or not config["study"]:
        raise ValueError(f"{source}: id and study must be non-empty")
    if not config["seeds"] or any(not isinstance(seed, int) for seed in config["seeds"]):
        raise ValueError(f"{source}: seeds must be a non-empty list of integers")
    for key in ("train_sigmas", "eval_sigmas"):
        values = config[key]
        if not values or any(not isinstance(value, (int, float)) or value < 0 for value in values):
            raise ValueError(f"{source}: {key} must contain non-negative numbers")
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["num_workers"] < 0:
        raise ValueError(f"{source}: epochs/batch_size must be positive and num_workers non-negative")
    if config["lr"] <= 0 or config["early_stopping_patience"] < 1:
        raise ValueError(f"{source}: invalid learning rate or early stopping patience")
    if config["mode"] == "cage":
        if "lambda1" not in config or "lambda2" not in config:
            raise ValueError(f"{source}: cage mode requires lambda1 and lambda2")
        if config["lambda1"] < 0 or config["lambda2"] < 0:
            raise ValueError(f"{source}: lambda values must be non-negative")


def resolved_config(config: dict, seed: int, run_dir: str | Path, cnn_checkpoint: str | None = None) -> dict:
    """Produce the exact Trainer config for one seed without mutating the source."""
    result = deepcopy(config)
    run_dir = Path(run_dir)
    result.update(
        {
            "seed": seed,
            "run_name": f"{config['id']}_seed{seed}",
            "noise_sigmas": list(config["train_sigmas"]),
            "metrics_dir": str(run_dir / "metrics"),
            "checkpoints_dir": str(run_dir / "checkpoints"),
            "plots_dir": str(run_dir / "plots"),
        }
    )
    if cnn_checkpoint is not None:
        result["cnn_checkpoint"] = cnn_checkpoint
    return result


def run_directory(output_root: str | Path, study: str, config_id: str, seed: int) -> Path:
    return Path(output_root) / study / "runs" / config_id / f"seed-{seed}"
