"""Command-line experiment runner for the CAGE v0.1 study.

Example: ``python -m cage.run train --config experiments/configs/v0.1.json``.
Run artifacts stay under ``runs/``; only derived release artifacts belong in
``results/v0.1``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from cage.artifacts import as_image, signed_difference
from cage.gradcam_analysis import gradcam_metrics, clean_reference
from cage.config import load_configs, resolved_config, run_directory
from cage.data import get_dataloaders
from cage.evaluate import evaluate_study_models, paired_gaussian_noise
from cage.gradcam import compute_gradcam
from cage.models.cnn import WeakCNN
from cage.models.unet import UNet
from cage.repro import configure_tf32, seed_everything, write_manifest
from cage.trainer import Trainer


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _checkpoint_for(run_dir: Path, config_id: str, seed: int) -> Path:
    return run_dir / "checkpoints" / f"{config_id}_seed{seed}.pt"


def _is_complete(run_dir: Path) -> bool:
    # The manifest is written only after training returns, while checkpoints
    # appear at every validation improvement, so only the manifest means done.
    return (run_dir / "manifest.json").exists()


def _selected_jobs(configs: list[dict], selected_ids: set[str] | None, selected_seed: int | None) -> list[tuple[dict, int]]:
    jobs = []
    for source in configs:
        if selected_ids and source["id"] not in selected_ids:
            continue
        for seed in source["seeds"]:
            if selected_seed is None or seed == selected_seed:
                jobs.append((source, seed))
    # Baselines first, since every denoiser needs the WeakCNN of its seed
    return sorted(jobs, key=lambda job: job[0]["mode"] != "cnn")


def _baseline_dir(output_root: str, source: dict, seed: int) -> Path:
    return run_directory(output_root, source["study"], "weakcnn", seed)


def _train_one(source: dict, seed: int, output_root: str) -> None:
    run_dir = run_directory(output_root, source["study"], source["id"], seed)
    checkpoint = _checkpoint_for(run_dir, source["id"], seed)
    cnn_checkpoint = None
    if source["mode"] != "cnn":
        baseline_dir = _baseline_dir(output_root, source, seed)
        cnn_checkpoint = _checkpoint_for(baseline_dir, "weakcnn", seed)
        if not _is_complete(baseline_dir):
            raise FileNotFoundError(
                f"{source['id']} seed={seed} requires a completed baseline run in {baseline_dir}"
            )
    config = resolved_config(source, seed, run_dir, str(cnn_checkpoint) if cnn_checkpoint else None)
    # An interrupted run left a checkpoint and a partial epoch log, so start it over
    stale = [checkpoint, Path(config["metrics_dir"]) / f"{config['run_name']}.json"]
    if any(path.exists() for path in stale):
        print(f"restarting interrupted run {source['id']} seed={seed}")
        for path in stale:
            path.unlink(missing_ok=True)
    state = seed_everything(seed)
    config["device"] = resolve_device(config["device"])
    Trainer(config).train()
    if not checkpoint.exists():
        raise RuntimeError(f"training {source['id']} seed={seed} finished without checkpoint")
    write_manifest(
        run_dir / "manifest.json", config=config, device=config["device"],
        checkpoint=checkpoint, extra={"seed_state": state},
    )


def train(
    config_path: str, output_root: str, selected_ids: set[str] | None, selected_seed: int | None,
    dry_run: bool, jobs: int = 1,
) -> None:
    pending = []
    for source, seed in _selected_jobs(load_configs(config_path), selected_ids, selected_seed):
        run_dir = run_directory(output_root, source["study"], source["id"], seed)
        if dry_run:
            print(f"would train {source['id']} seed={seed} -> {run_dir}")
        elif _is_complete(run_dir):
            print(f"skipping {source['id']} seed={seed}; already complete")
        else:
            pending.append((source, seed))
    if dry_run or not pending:
        return
    if jobs <= 1:
        for source, seed in pending:
            _train_one(source, seed, output_root)
        return
    _train_parallel(config_path, output_root, pending, jobs)


def _job_command(config_path: str, output_root: str, config_id: str, seed: int) -> list[str]:
    return [
        sys.executable, "-u", "-m", "cage.run", "train", "--config", config_path,
        "--output-root", output_root, "--run-id", config_id, "--seed", str(seed),
    ]


def _relay_output(process: subprocess.Popen, log_path: Path, run_name: str) -> None:
    """Copy one job's output to its log file and to the console.

    Trainer status lines already start with the run name; anything else, such
    as a warning or traceback, gets the run name added so it stays attributable.
    """
    assert process.stdout is not None
    with open(log_path, "a") as log:
        for line in process.stdout:
            log.write(line)
            log.flush()
            prefix = "" if line.startswith(f"{run_name}:") else f"{run_name}: "
            print(prefix + line, end="", flush=True)


def _train_parallel(config_path: str, output_root: str, pending: list[tuple[dict, int]], jobs: int) -> None:
    """Run each (config, seed) in its own process, at most ``jobs`` at a time.

    Separate processes keep seeding, CUDA state and failures independent, so a
    run trains exactly as it would serially. A denoiser waits until the
    WeakCNN of its seed is complete; if that baseline fails, it is skipped.
    """
    # Download once here, because concurrent first downloads would write the same file
    get_dataloaders(data_dir=pending[0][0]["data_dir"], num_workers=0)
    queue = list(pending)
    running: dict[str, tuple[subprocess.Popen, threading.Thread, tuple[dict, int]]] = {}
    failed: list[str] = []
    skipped: list[str] = []
    failed_baselines: set[int] = set()
    started_at = time.monotonic()
    total = len(pending)
    finished = 0

    def label_of(source: dict, seed: int) -> str:
        return f"{source['id']} seed={seed}"

    def ready(source: dict, seed: int) -> bool:
        return source["mode"] == "cnn" or _is_complete(_baseline_dir(output_root, source, seed))

    try:
        while queue or running:
            for label, (process, relay, (source, seed)) in list(running.items()):
                if process.poll() is None:
                    continue
                relay.join()
                del running[label]
                finished += 1
                elapsed = (time.monotonic() - started_at) / 60
                if process.returncode == 0:
                    print(f"done {label} ({finished}/{total}, {elapsed:.1f} min elapsed)", flush=True)
                else:
                    log = run_directory(output_root, source["study"], source["id"], seed) / "train.log"
                    print(f"FAILED {label} with exit code {process.returncode}; see {log}", flush=True)
                    failed.append(label)
                    if source["mode"] == "cnn":
                        failed_baselines.add(seed)

            for source, seed in list(queue):
                if source["mode"] != "cnn" and seed in failed_baselines:
                    queue.remove((source, seed))
                    skipped.append(label_of(source, seed))
                    finished += 1
                    print(f"skipping {label_of(source, seed)}; its WeakCNN baseline failed", flush=True)
                    continue
                if len(running) >= jobs or not ready(source, seed):
                    continue
                queue.remove((source, seed))
                run_dir = run_directory(output_root, source["study"], source["id"], seed)
                run_dir.mkdir(parents=True, exist_ok=True)
                label = label_of(source, seed)
                process = subprocess.Popen(
                    _job_command(config_path, output_root, source["id"], seed),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                )
                relay = threading.Thread(
                    target=_relay_output, args=(process, run_dir / "train.log", f"{source['id']}_seed{seed}"), daemon=True
                )
                relay.start()
                running[label] = (process, relay, (source, seed))
                print(f"started {label} ({len(running)} running, {len(queue)} queued)", flush=True)

            if running:
                time.sleep(1.0)
            elif queue:
                # Nothing running and nothing startable means a baseline is missing
                for source, seed in queue:
                    skipped.append(label_of(source, seed))
                    print(f"skipping {label_of(source, seed)}; no completed WeakCNN baseline", flush=True)
                queue.clear()
    except KeyboardInterrupt:
        for process, _, _ in running.values():
            process.terminate()
        for process, _, _ in running.values():
            process.wait()
        raise

    print(f"parallel training finished: {total - len(failed) - len(skipped)} complete, "
          f"{len(failed)} failed, {len(skipped)} skipped")
    if failed or skipped:
        raise RuntimeError(f"failed: {failed}; skipped: {skipped}")


def _study_manifests(study_dir: Path) -> list[dict]:
    manifests = []
    for path in sorted((study_dir / "runs").glob("**/manifest.json")):
        manifests.append(json.loads(path.read_text()))
    return manifests


def evaluate(study: str, output_root: str, evaluation_seed: int, dry_run: bool) -> None:
    study_dir = Path(output_root) / study
    manifests = _study_manifests(study_dir)
    if not manifests:
        raise FileNotFoundError(f"no completed manifests found under {study_dir / 'runs'}")
    # Test evaluation is strict FP32 whatever precision training used
    configure_tf32(False)
    per_seed: dict[int, list[dict]] = defaultdict(list)
    for manifest in manifests:
        per_seed[int(manifest["config"]["seed"])].append(manifest)
    for seed, entries in sorted(per_seed.items()):
        baseline = next((entry for entry in entries if entry["config"]["mode"] == "cnn"), None)
        if baseline is None:
            raise RuntimeError(f"seed {seed} has denoisers but no WeakCNN baseline")
        models = {
            entry["config"]["id"]: entry["checkpoint"]
            for entry in entries if entry["config"]["mode"] != "cnn"
        }
        target = study_dir / "evaluations" / f"seed-{seed}.json"
        if dry_run:
            print(f"would evaluate seed={seed}, {len(models)} denoisers -> {target}")
            continue
        config = baseline["config"]
        results = evaluate_study_models(
            cnn_checkpoint=baseline["checkpoint"], model_checkpoints=models,
            sigmas=config["eval_sigmas"], evaluation_seed=evaluation_seed,
            data_dir=config["data_dir"], batch_size=config["batch_size"],
            num_workers=config["num_workers"], device_str=config["device"],
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"seed": seed, "evaluation_seed": evaluation_seed, "results": results}, indent=2) + "\n")
        print(f"evaluated seed={seed}: {target}")


def _t_critical_95(n: int) -> float | None:
    # Two-sided Student-t 97.5% quantiles for small experiment counts.
    return {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(n, 1.96 if n > 30 else 2.262)


def _summary(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return {"n": n, "mean": mean, "ci95": None}
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return {"n": n, "mean": mean, "ci95": _t_critical_95(n) * math.sqrt(variance / n)}


def aggregate(study: str, output_root: str, results_root: str) -> None:
    study_dir = Path(output_root) / study
    evaluations = [json.loads(path.read_text()) for path in sorted((study_dir / "evaluations").glob("seed-*.json"))]
    if not evaluations:
        raise FileNotFoundError(f"no evaluations found under {study_dir / 'evaluations'}")
    metadata = {
        manifest["config"]["id"]: manifest["config"]
        for manifest in _study_manifests(study_dir)
    }
    rows: list[dict] = []
    for evaluation in evaluations:
        for sigma, conditions in evaluation["results"].items():
            for condition, metrics in conditions.items():
                config = metadata.get(condition, {})
                rows.append({
                    "seed": evaluation["seed"], "test_sigma": float(sigma), "condition": condition,
                    "accuracy": metrics["accuracy"], "ssim": metrics["ssim"],
                    "fidelity": config.get("fidelity", "baseline"), "role": config.get("role", condition),
                    "lambda1": config.get("lambda1", ""), "lambda2": config.get("lambda2", ""),
                    "train_sigmas": ";".join(map(str, config.get("train_sigmas", []))),
                })
    destination = Path(results_root) / study
    destination.mkdir(parents=True, exist_ok=True)
    with open(destination / "per_seed_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        for metric in ("accuracy", "ssim"):
            groups[(row["condition"], row["fidelity"], row["role"], row["test_sigma"], metric)].append(float(row[metric]))
    training = {row["condition"]: {"lambda1": row["lambda1"], "train_sigmas": row["train_sigmas"]} for row in rows}
    summaries = [
        {"condition": key[0], "fidelity": key[1], "role": key[2], "test_sigma": key[3], "metric": key[4], **training[key[0]], **_summary(values)}
        for key, values in sorted(groups.items())
    ]

    controls = {(row["seed"], row["fidelity"], row["test_sigma"]): row for row in rows if row["role"] == "control"}
    deltas: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        control = controls.get((row["seed"], row["fidelity"], row["test_sigma"]))
        if control is not None and row["condition"] != control["condition"]:
            deltas[(row["condition"], row["fidelity"], row["test_sigma"], "accuracy")].append(float(row["accuracy"]) - float(control["accuracy"]))
            deltas[(row["condition"], row["fidelity"], row["test_sigma"], "ssim")].append(float(row["ssim"]) - float(control["ssim"]))
    paired = [
        {"condition": key[0], "fidelity": key[1], "test_sigma": key[2], "metric": key[3], **_summary(values)}
        for key, values in sorted(deltas.items())
    ]
    (destination / "summary.json").write_text(json.dumps({"summary": summaries, "paired_deltas_vs_control": paired}, indent=2) + "\n")
    print(f"wrote aggregate artifacts to {destination}")


def _local_checkpoint(output_root: str, study: str, config: dict) -> Path:
    # Manifests record where training saved the checkpoint (a Colab path for
    # the campaign), so analysis looks it up under this machine's output root.
    return _checkpoint_for(run_directory(output_root, study, config["id"], config["seed"]), config["id"], config["seed"])


def _qualitative_figure(study: str, output_root: str, results_root: str, evaluation_seed: int = 20260715) -> None:
    """Render fixed, class-stratified GradCAM/difference-map examples when runs exist."""
    manifests = _study_manifests(Path(output_root) / study)
    by_key = {(entry["config"]["id"], entry["config"]["seed"]): entry for entry in manifests}
    seed = min((entry["config"]["seed"] for entry in manifests), default=None)
    required = ("weakcnn", "mse_control", "mse_main", "msssim_control", "msssim_main")
    if seed is None or any((name, seed) not in by_key for name in required):
        print("qualitative analysis skipped: complete seed-17 control/main checkpoints are not available")
        return
    entries = {name: by_key[(name, seed)] for name in required}
    config = entries["weakcnn"]["config"]
    device = torch.device(resolve_device("auto"))
    cnn = WeakCNN().to(device)
    cnn.load_state_dict(torch.load(_local_checkpoint(output_root, study, config), map_location=device))
    cnn.freeze()
    models = {}
    for name in required[1:]:
        model = UNet().to(device)
        model.load_state_dict(torch.load(_local_checkpoint(output_root, study, entries[name]["config"]), map_location=device))
        model.eval()
        models[name] = model
    dataset = get_dataloaders(data_dir=config["data_dir"], batch_size=config["batch_size"], num_workers=0)["test"].dataset
    selected, seen = [], set()
    for index in range(len(dataset)):
        _, label = dataset[index]
        label = int(label)
        if label not in seen:
            selected.append(index)
            seen.add(label)
        if len(selected) == 5:
            break
    clean = torch.stack([dataset[index][0] for index in selected]).to(device)
    noisy = paired_gaussian_noise(clean.cpu(), torch.tensor(selected), 0.3, evaluation_seed).to(device)
    with torch.no_grad():
        outputs = {name: model(noisy) for name, model in models.items()}
    # GradCAM intentionally targets each input's predicted class. These are
    # representative qualitative diagnostics, not an alignment statistic.
    cams = {"noisy": compute_gradcam(cnn, noisy, cnn(noisy).argmax(1).tolist())}
    for name, output in outputs.items():
        cams[name] = compute_gradcam(cnn, output, cnn(output).argmax(1).tolist())
    columns = [("Clean", clean, None), ("Noisy", noisy, None), ("GradCAM noisy", None, cams["noisy"])]
    for prefix, display in (("mse_control", "MSE control"), ("mse_main", "CAGE MSE"), ("msssim_control", "MS-SSIM control"), ("msssim_main", "CAGE MS-SSIM")):
        columns.append((display, outputs[prefix], None))
        if prefix.endswith("main"):
            columns.append((f"Diff {display}−clean", outputs[prefix] - clean, "diff"))
        columns.append((f"GradCAM {display}", None, cams[prefix]))
    fig, axes = plt.subplots(len(selected), len(columns), figsize=(2.1 * len(columns), 2.4 * len(selected)))
    for col, (title, _, _) in enumerate(columns):
        axes[0, col].set_title(title, fontsize=8)
    for row in range(len(selected)):
        for col, (_, images, cam) in enumerate(columns):
            ax = axes[row, col]
            if isinstance(cam, str) and cam == "diff":
                image = signed_difference(images[row])
                limit = abs(image).max() + 1e-6
                ax.imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit)
            elif images is not None:
                ax.imshow(as_image(images[row]))
            else:
                ax.imshow(cam[row], cmap="jet", vmin=0, vmax=1)
            ax.axis("off")
    fig.suptitle("Representative, class-stratified CAGE diagnostics (seed 17, test σ=0.3)", fontsize=12)
    fig.tight_layout()
    destination = Path(results_root) / study / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "qualitative_gradcam_difference_sigma0.3.png", dpi=150)
    plt.close(fig)


FIDELITY_LABELS = {"mse": "MSE", "msssim": "MS-SSIM"}


def _pareto_figure(summary: dict, out: Path) -> None:
    """Accuracy against SSIM at test sigma 0.3, one curve per fidelity term from lambda1=0 to the control."""
    fig, ax = plt.subplots(figsize=(7, 5))
    points = []
    for fidelity, label in FIDELITY_LABELS.items():
        # The control is the lambda1=1 end of each sweep.
        branch = sorted(
            (1.0 if row["role"] == "control" else float(row["lambda1"]), row["condition"])
            for row in summary.values()
            if row["fidelity"] == fidelity and row["role"] in {"control", "lambda", "main"} and row["test_sigma"] == 0.3 and row["metric"] == "accuracy"
        )
        if not branch:
            continue
        acc = [summary[(name, 0.3, "accuracy")] for _, name in branch]
        ssim = [summary[(name, 0.3, "ssim")] for _, name in branch]
        line = ax.errorbar(
            [row["mean"] for row in ssim], [row["mean"] for row in acc],
            xerr=[row["ci95"] or 0 for row in ssim], yerr=[row["ci95"] or 0 for row in acc],
            marker="o", capsize=2, label=label,
        )
        for (lambda1, _), x, y in zip(branch, ssim, acc):
            ax.annotate(f"{lambda1:g}", (x["mean"], y["mean"]), textcoords="offset points", xytext=(5, -11), fontsize=8, color=line[0].get_color())
            points.append((x["mean"], y["mean"]))
    front = sorted(p for p in points if not any(q[0] >= p[0] and q[1] >= p[1] and q != p for q in points))
    if front:
        ax.plot(*zip(*front), color="0.5", lw=6, alpha=0.25, zorder=0, label="Pareto front")
    # Noisy input (about 0.2 accuracy) is left out so the front stays readable.
    if ("clean", 0.3, "accuracy") in summary:
        ax.axhline(summary[("clean", 0.3, "accuracy")]["mean"], color="0.3", ls="--", lw=1, label="clean input")
    ax.set_xlabel("SSIM to clean image")
    ax.set_ylabel("WeakCNN test accuracy")
    ax.set_title(r"Accuracy vs. fidelity at test $\sigma$=0.3 (labels: $\lambda_1$)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pareto_sigma0.3.png", dpi=150)
    plt.close(fig)


def _sigma_figure(summary: dict, out: Path) -> None:
    """Test accuracy against test sigma for each training-noise schedule, one panel per fidelity term."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (fidelity, label) in zip(axes, FIDELITY_LABELS.items()):
        conditions = sorted({
            (row["train_sigmas"], row["condition"]) for row in summary.values()
            if row["fidelity"] == fidelity and row["role"] in {"main", "sigma"}
        })
        for train_sigmas, name in conditions:
            series = sorted((key[1], row["mean"]) for key, row in summary.items() if key[0] == name and key[2] == "accuracy")
            ax.plot(*zip(*series), marker="o", label=f"train {train_sigmas.replace(';', ', ')}")
        for name, style in (("clean", "--"), ("noisy", ":")):
            series = sorted((key[1], row["mean"]) for key, row in summary.items() if key[0] == name and key[2] == "accuracy")
            if series:
                ax.plot(*zip(*series), color="0.3", ls=style, lw=1, label=f"{name} input")
        ax.set_title(f"CAGE {label}")
        ax.set_xlabel(r"test $\sigma$")
    axes[0].set_ylabel("WeakCNN test accuracy")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sigma_generalization.png", dpi=150)
    plt.close(fig)


def _gradcam_cases(manifests: list[dict]) -> dict[str, float]:
    """Denoisers in the GradCAM analysis and the test sigma each is probed at."""
    cases = {}
    for entry in manifests:
        config = entry["config"]
        if config.get("role") in {"control", "main", "lambda"}:
            cases[config["id"]] = 0.3
        elif config.get("role") == "sigma" and config.get("train_sigmas") == [0.0]:
            # Trained without noise, so probed on clean input where the edit is all addition
            cases[config["id"]] = 0.0
    return cases


def _paste_figure(summary: dict, out: Path) -> None:
    conditions = list(summary)
    bars = (("none", "clean image"), ("top", "edit inside its own top-20% GradCAM region"), ("shuffled", "edit inside another image's top-20% region"), ("all", "full edit"))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / len(bars)
    for offset, (key, label) in enumerate(bars):
        rows = [summary[name][f"accuracy_paste_{key}"] for name in conditions]
        xs = [index + (offset - (len(bars) - 1) / 2) * width for index in range(len(conditions))]
        ax.bar(xs, [row["mean"] for row in rows], width, yerr=[row["ci95"] or 0 for row in rows], capsize=2, label=label)
    ax.set_xticks(range(len(conditions)), [name.replace("_lambda_075_025", " λ1=0.75").replace("_sigma_000", " σ=0").replace("_", " ") for name in conditions], rotation=20, ha="right")
    ax.set_ylim(0.5, None)
    ax.set_ylabel("WeakCNN test accuracy")
    ax.set_title("Pasting each denoiser's edit back onto the clean image")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "gradcam_paste_test.png", dpi=150)
    plt.close(fig)


def _gradcam_study(study: str, output_root: str, results_root: str, evaluation_seed: int = 20260715) -> None:
    """Measure where WeakCNN looks on each denoiser's output, per seed, and summarize across seeds."""
    manifests = _study_manifests(Path(output_root) / study)
    cases = _gradcam_cases(manifests)
    per_seed: dict[int, list[dict]] = defaultdict(list)
    for entry in manifests:
        per_seed[int(entry["config"]["seed"])].append(entry)
    usable = {
        seed: entries for seed, entries in per_seed.items()
        if all(_local_checkpoint(output_root, study, entry["config"]).exists() for entry in entries)
        and any(entry["config"]["mode"] == "cnn" for entry in entries)
    }
    if not cases or not usable:
        print("GradCAM analysis skipped: no seed has a WeakCNN and denoiser checkpoints")
        return
    configure_tf32(False)
    device = torch.device(resolve_device("auto"))
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for seed, entries in sorted(usable.items()):
        baseline = next(entry["config"] for entry in entries if entry["config"]["mode"] == "cnn")
        cnn = WeakCNN().to(device)
        cnn.load_state_dict(torch.load(_local_checkpoint(output_root, study, baseline), map_location=device))
        cnn.freeze()
        dataset = get_dataloaders(data_dir=baseline["data_dir"], batch_size=baseline["batch_size"], num_workers=0)["test"].dataset
        reference = clean_reference(cnn, dataset, baseline["batch_size"], device)
        for entry in entries:
            config = entry["config"]
            if config["id"] not in cases:
                continue
            denoiser = UNet().to(device)
            denoiser.load_state_dict(torch.load(_local_checkpoint(output_root, study, config), map_location=device))
            denoiser.eval()
            metrics = gradcam_metrics(cnn, denoiser, dataset, reference, cases[config["id"]], evaluation_seed, baseline["batch_size"], device)
            for name, value in metrics.items():
                values[config["id"]][name].append(value)
            print(f"GradCAM analysis seed={seed} {config['id']}")
    summary = {condition: {name: _summary(series) for name, series in metrics.items()} for condition, metrics in sorted(values.items())}
    destination = Path(results_root) / study
    payload = {"evaluation_seed": evaluation_seed, "test_sigma": {name: cases[name] for name in summary}, "per_seed": values, "summary": summary}
    (destination / "gradcam.json").write_text(json.dumps(payload, indent=2) + "\n")
    _paste_figure(summary, destination / "figures")
    print(f"wrote {destination / 'gradcam.json'}")


def analyze(study: str, results_root: str, output_root: str = "runs") -> None:
    """Create the Pareto and sigma figures, the GradCAM analysis and qualitative diagnostics."""
    source = Path(results_root) / study / "summary.json"
    payload = json.loads(source.read_text())
    summary = {(row["condition"], row["test_sigma"], row["metric"]): row for row in payload["summary"]}
    out = source.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    _pareto_figure(summary, out)
    _sigma_figure(summary, out)
    print(f"wrote figures to {out}")
    _gradcam_study(study, output_root, results_root)
    _qualitative_figure(study, output_root, results_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible CAGE study runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--output-root", default="runs")
    train_parser.add_argument("--run-id", action="append", dest="run_ids")
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.add_argument(
        "--jobs", type=int, default=1,
        help="training processes to run at once; each (config, seed) gets its own process",
    )
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--study", required=True)
    eval_parser.add_argument("--output-root", default="runs")
    eval_parser.add_argument("--evaluation-seed", type=int, default=20260715)
    eval_parser.add_argument("--dry-run", action="store_true")
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--study", required=True)
    aggregate_parser.add_argument("--output-root", default="runs")
    aggregate_parser.add_argument("--results-root", default="results")
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--study", required=True)
    analyze_parser.add_argument("--results-root", default="results")
    analyze_parser.add_argument("--output-root", default="runs")
    args = parser.parse_args()
    if args.command == "train":
        train(args.config, args.output_root, set(args.run_ids) if args.run_ids else None, args.seed,
              args.dry_run, jobs=args.jobs)
    elif args.command == "evaluate":
        evaluate(args.study, args.output_root, args.evaluation_seed, args.dry_run)
    elif args.command == "aggregate":
        aggregate(args.study, args.output_root, args.results_root)
    else:
        analyze(args.study, args.results_root, args.output_root)


if __name__ == "__main__":
    main()
