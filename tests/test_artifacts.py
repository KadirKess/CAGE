import json
import os
import pytest
from cage.artifacts import append_epoch_log, save_loss_curves, save_accuracy_curves


@pytest.fixture
def dirs(tmp_path):
    return str(tmp_path / "metrics"), str(tmp_path / "plots")


def test_append_creates_json_file(dirs):
    metrics_dir, _ = dirs
    append_epoch_log(
        "run1",
        {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4},
        metrics_dir=metrics_dir,
    )
    assert os.path.exists(os.path.join(metrics_dir, "run1.json"))


def test_append_accumulates_entries(dirs):
    metrics_dir, _ = dirs
    for i in range(3):
        append_epoch_log(
            "run2",
            {"epoch": i + 1, "train_loss": 0.1 * i, "val_loss": 0.1 * i},
            metrics_dir=metrics_dir,
        )
    with open(os.path.join(metrics_dir, "run2.json")) as f:
        data = json.load(f)
    assert len(data) == 3
    assert data[2]["epoch"] == 3


def test_save_loss_curves_creates_png(dirs):
    metrics_dir, plots_dir = dirs
    logs = [
        {"epoch": i, "train_loss": 0.5 - i * 0.05, "val_loss": 0.6 - i * 0.05}
        for i in range(1, 6)
    ]
    save_loss_curves("run3", logs, plots_dir=plots_dir)
    assert os.path.exists(os.path.join(plots_dir, "run3_loss_curves.png"))


def test_save_accuracy_curves_creates_png(dirs):
    metrics_dir, plots_dir = dirs
    logs = [
        {
            "epoch": i,
            "train_loss": 0.1,
            "val_loss": 0.1,
            "cnn_accuracy_per_sigma": {"0.1": 0.9, "0.3": 0.8, "0.5": 0.7},
        }
        for i in range(1, 4)
    ]
    save_accuracy_curves("run4", logs, plots_dir=plots_dir)
    assert os.path.exists(os.path.join(plots_dir, "run4_accuracy_curves.png"))
