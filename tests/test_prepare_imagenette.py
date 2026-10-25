import importlib.util
import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_imagenette.py"
_spec = importlib.util.spec_from_file_location("prepare_imagenette", _SCRIPT)
prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare)


def _jpeg(width: int, height: int, color: tuple[int, int, int], mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (width, height), color if mode == "RGB" else color[0]).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def archive(tmp_path):
    """A miniature imagenette2-160.tgz with two classes per split."""
    path = tmp_path / "imagenette2-160.tgz"
    wnids = list(prepare.CLASSES)
    with tarfile.open(path, "w:gz") as tar:
        files = [
            ("train", wnids[0], "a.JPEG", _jpeg(240, 160, (255, 0, 0))),
            ("train", wnids[3], "b.JPEG", _jpeg(160, 213, (0, 0, 255))),
            # Some Imagenette images are grayscale and must come out RGB
            ("train", wnids[3], "c.JPEG", _jpeg(160, 160, (90, 90, 90), mode="L")),
        ] + [("val", wnids[i % 2], f"v{i}.JPEG", _jpeg(200, 160, (0, 255, 0))) for i in range(8)]
        for split, wnid, name, data in files:
            info = tarfile.TarInfo(f"imagenette2-160/{split}/{wnid}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_archive_becomes_rgb_128_crops_with_class_indices(archive):
    splits = prepare.read_archive(archive, 128)
    images, labels = splits["train"]
    assert images.shape == (3, 128, 128, 3) and images.dtype == np.uint8
    assert sorted(labels.tolist()) == [0, 3, 3]
    red = images[labels.tolist().index(0)]
    assert red[..., 0].mean() > 200 and red[..., 2].mean() < 50
    assert splits["val"][0].shape == (8, 128, 128, 3)


def test_shorter_side_is_resized_then_center_cropped():
    # A 2:1 image with a black left half and a white right half
    image = Image.new("RGB", (320, 160))
    image.paste((255, 255, 255), (160, 0, 320, 160))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    crop = prepare.resize_and_crop(buffer.getvalue(), 128)
    assert crop.shape == (128, 128, 3)
    # The center crop straddles the boundary, so its left half is dark and its right half bright
    assert crop[:, :50].mean() < 30 and crop[:, -50:].mean() > 225


def test_val_split_halves_each_class_without_overlap():
    labels = np.array([0] * 7 + [1] * 4 + [2] * 9)
    first, second = prepare.split_per_class(labels, seed=0)
    assert set(first).isdisjoint(second) and len(first) + len(second) == len(labels)
    for label in (0, 1, 2):
        in_first = (labels[first] == label).sum()
        in_second = (labels[second] == label).sum()
        assert abs(int(in_first) - int(in_second)) <= 1
    again, _ = prepare.split_per_class(labels, seed=0)
    assert np.array_equal(first, again)


def test_main_writes_npz_from_a_local_archive(archive, tmp_path, monkeypatch):
    output = tmp_path / "out" / "imagenette_128.npz"
    monkeypatch.setattr("sys.argv", ["prepare", "--archive", str(archive), "--output", str(output)])
    assert prepare.main() == 0
    with np.load(output) as data:
        assert data["train_images"].shape == (3, 128, 128, 3)
        assert len(data["val_labels"]) + len(data["test_labels"]) == 8
        assert data["class_names"][0] == "tench"
        assert len(str(data["source_sha256"])) == 64
