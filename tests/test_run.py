import json
import sys
import textwrap

import pytest

import cage.run as run_module
from cage.config import expand_study_config, load_configs, run_directory
from cage.run import aggregate, analyze, evaluate, train

CONFIG = "experiments/configs/v0.1.json"


class _FakeTrainer:
    """Stands in for Trainer: writes a checkpoint and one epoch log row."""

    calls: list[str] = []

    def __init__(self, config):
        self.config = config

    def train(self):
        from pathlib import Path

        _FakeTrainer.calls.append(self.config["run_name"])
        Path(self.config["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(self.config["checkpoints_dir"]) / f"{self.config['run_name']}.pt").write_bytes(b"new")
        Path(self.config["metrics_dir"]).mkdir(parents=True, exist_ok=True)
        log = Path(self.config["metrics_dir"]) / f"{self.config['run_name']}.json"
        rows = json.loads(log.read_text()) if log.exists() else []
        log.write_text(json.dumps(rows + [{"epoch": len(rows) + 1}]))
        return []


@pytest.fixture
def fake_trainer(monkeypatch):
    _FakeTrainer.calls = []
    monkeypatch.setattr(run_module, "Trainer", _FakeTrainer)
    monkeypatch.setattr(run_module, "resolve_device", lambda value: "cpu")
    return _FakeTrainer


def test_interrupted_run_with_checkpoint_but_no_manifest_restarts(tmp_path, fake_trainer):
    run_dir = run_directory(tmp_path, "v0.1", "weakcnn", 17)
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "weakcnn_seed17.pt").write_bytes(b"stale")
    (run_dir / "metrics").mkdir()
    (run_dir / "metrics" / "weakcnn_seed17.json").write_text(json.dumps([{"epoch": 1}, {"epoch": 2}]))
    train(CONFIG, str(tmp_path), {"weakcnn"}, 17, dry_run=False)
    assert fake_trainer.calls == ["weakcnn_seed17"]
    assert (run_dir / "manifest.json").exists()
    # The partial log from the interrupted run must not be extended
    assert json.loads((run_dir / "metrics" / "weakcnn_seed17.json").read_text()) == [{"epoch": 1}]


def test_completed_run_is_skipped(tmp_path, fake_trainer, capsys):
    run_dir = run_directory(tmp_path, "v0.1", "weakcnn", 17)
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}")
    train(CONFIG, str(tmp_path), {"weakcnn"}, 17, dry_run=False)
    assert fake_trainer.calls == []
    assert "already complete" in capsys.readouterr().out


def test_denoiser_requires_completed_baseline_not_just_its_checkpoint(tmp_path, fake_trainer):
    baseline = run_directory(tmp_path, "v0.1", "weakcnn", 17)
    (baseline / "checkpoints").mkdir(parents=True)
    (baseline / "checkpoints" / "weakcnn_seed17.pt").write_bytes(b"partial")
    with pytest.raises(FileNotFoundError, match="completed baseline"):
        train(CONFIG, str(tmp_path), {"mse_main"}, 17, dry_run=False)


# A stand-in for one `cage.run train` job. It fails if a denoiser starts
# before its baseline manifest exists, and fails on purpose for seed 43's
# baseline when FAIL_BASELINE is set.
_FAKE_JOB = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    root, config_id, seed, fail = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
    run_dir = Path(root) / "v0.1" / "runs" / config_id / f"seed-{seed}"
    baseline = Path(root) / "v0.1" / "runs" / "weakcnn" / f"seed-{seed}" / "manifest.json"
    if config_id != "weakcnn" and not baseline.exists():
        sys.exit(3)
    if fail and config_id == "weakcnn" and seed == 43:
        sys.exit(1)
    print(f"{config_id}_seed{seed}: epoch 1/1 val=0.1", flush=True)
    print("a warning from some library", flush=True)
    time.sleep(0.2)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text("{}")
""")


@pytest.fixture
def fake_jobs(monkeypatch):
    monkeypatch.setattr(run_module, "get_dataloaders", lambda **kwargs: None)

    def use(fail_baseline: bool):
        monkeypatch.setattr(
            run_module, "_job_command",
            lambda config_path, output_root, config_id, seed: [
                sys.executable, "-c", _FAKE_JOB, output_root, config_id, str(seed), "1" if fail_baseline else "0",
            ],
        )
    return use


def test_parallel_training_runs_baselines_before_their_denoisers(tmp_path, fake_jobs, capsys):
    fake_jobs(fail_baseline=False)
    train(CONFIG, str(tmp_path), {"weakcnn", "mse_main", "msssim_control"}, None, dry_run=False, jobs=3)
    manifests = list((tmp_path / "v0.1" / "runs").glob("*/seed-*/manifest.json"))
    assert len(manifests) == 9
    output = capsys.readouterr().out
    assert "mse_main_seed17: epoch 1/1 val=0.1" in output
    # Lines that do not come from the trainer still say which run printed them
    assert "mse_main_seed17: a warning from some library" in output
    assert "9 complete, 0 failed, 0 skipped" in output
    log = run_directory(tmp_path, "v0.1", "mse_main", 17) / "train.log"
    assert "mse_main_seed17: epoch 1/1 val=0.1" in log.read_text()


def test_parallel_training_skips_denoisers_of_a_failed_baseline(tmp_path, fake_jobs, capsys):
    fake_jobs(fail_baseline=True)
    with pytest.raises(RuntimeError, match="weakcnn seed=43"):
        train(CONFIG, str(tmp_path), {"weakcnn", "mse_main"}, None, dry_run=False, jobs=2)
    output = capsys.readouterr().out
    assert "skipping mse_main seed=43; its WeakCNN baseline failed" in output
    assert (run_directory(tmp_path, "v0.1", "mse_main", 17) / "manifest.json").exists()
    assert not (run_directory(tmp_path, "v0.1", "mse_main", 43) / "manifest.json").exists()


def test_v01_matrix_expands_to_21_unique_configs():
    configs = load_configs("experiments/configs/v0.1.json")
    assert len(configs) == 21
    assert len({config["id"] for config in configs}) == 21
    assert all(config["seeds"] == [17, 43, 71] for config in configs)


def test_invalid_study_config_is_rejected():
    try:
        expand_study_config({"study": "x", "seeds": []})
    except ValueError as error:
        assert "eval_sigmas" in str(error)
    else:
        raise AssertionError("invalid study should fail")


def test_train_dry_run_never_creates_run_artifacts(tmp_path, capsys):
    train("experiments/configs/v0.1.json", str(tmp_path), {"weakcnn"}, 17, dry_run=True)
    assert "would train weakcnn seed=17" in capsys.readouterr().out
    assert not list(tmp_path.rglob("manifest.json"))


def test_aggregate_writes_paired_delta(tmp_path):
    study_dir = tmp_path / "runs" / "v0.1"
    for seed, control, main in ((17, 0.5, 0.7), (43, 0.6, 0.8), (71, 0.7, 0.9)):
        for condition, role in (("mse_control", "control"), ("mse_main", "main")):
            path = study_dir / "runs" / condition / f"seed-{seed}" / "manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"config": {"id": condition, "role": role, "fidelity": "mse"}}))
        payload = {"seed": seed, "results": {"0.3": {
            "mse_control": {"accuracy": control, "ssim": 0.9},
            "mse_main": {"accuracy": main, "ssim": 0.8},
        }}}
        target = study_dir / "evaluations" / f"seed-{seed}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload))
    aggregate("v0.1", str(tmp_path / "runs"), str(tmp_path / "results"))
    summary = json.loads((tmp_path / "results" / "v0.1" / "summary.json").read_text())
    accuracy_delta = next(row for row in summary["paired_deltas_vs_control"] if row["metric"] == "accuracy")
    assert accuracy_delta["n"] == 3
    assert round(accuracy_delta["mean"], 6) == 0.2


def test_evaluate_dry_run_reads_manifests_without_models(tmp_path, capsys):
    path = tmp_path / "runs" / "v0.1" / "runs" / "weakcnn" / "seed-17" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"checkpoint": "unused.pt", "config": {"id": "weakcnn", "seed": 17, "mode": "cnn"}}))
    evaluate("v0.1", str(tmp_path / "runs"), evaluation_seed=1, dry_run=True)
    assert "would evaluate seed=17" in capsys.readouterr().out


def test_analyze_writes_frontier_and_skips_missing_checkpoints(tmp_path):
    destination = tmp_path / "results" / "v0.1"
    destination.mkdir(parents=True)
    payload = {
        "summary": [
            {"condition": name, "fidelity": "mse", "role": role, "lambda1": lambda1, "train_sigmas": "0.3", "test_sigma": 0.3, "metric": metric, "mean": mean, "ci95": 0.01}
            for name, role, lambda1 in (("mse_control", "control", ""), ("mse_main", "main", 0.5))
            for metric, mean in (("accuracy", 0.7), ("ssim", 0.5))
        ],
        "paired_deltas_vs_control": []
    }
    (destination / "summary.json").write_text(json.dumps(payload))
    analyze("v0.1", str(tmp_path / "results"), str(tmp_path / "runs"))
    assert (destination / "figures" / "pareto_sigma0.3.png").exists()
    assert (destination / "figures" / "sigma_generalization.png").exists()
