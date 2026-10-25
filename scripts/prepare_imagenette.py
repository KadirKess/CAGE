"""Download Imagenette, resize it to 128x128 RGB, and save one compressed npz.

uv run python scripts/prepare_imagenette.py [--output data/imagenette_128.npz]

Starts from the 160px release (shorter side 160). Each image is converted to
RGB, its shorter side resized to 128 with bicubic filtering, and center
cropped to 128x128. The official train split is kept for training. The
official val split has no separate test set, so it is split per class into
val and test halves with a fixed seed. The npz holds uint8 HWC images, int64
labels, class names and provenance; cage/data.py reads it.
"""

import argparse
import hashlib
import io
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
ARCHIVE_ROOT = "imagenette2-160"
SPLIT_SEED = 20261025
# Sorted WordNet ids, so class index i is the i-th directory name
CLASSES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def download(url: str, destination: Path) -> str:
    digest = hashlib.sha256()
    started = time.monotonic()
    with urlopen(url, timeout=60) as source, destination.open("wb") as output:
        total = int(source.headers.get("Content-Length") or 0)
        done, reported = 0, 0
        while block := source.read(1 << 20):
            output.write(block)
            digest.update(block)
            done += len(block)
            if total and done * 10 // total > reported:
                reported = done * 10 // total
                print(f"download: {done / 1e6:.0f}/{total / 1e6:.0f} MB, "
                      f"{time.monotonic() - started:.0f}s", flush=True)
    return digest.hexdigest()


def resize_and_crop(data: bytes, size: int) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    width, height = image.size
    scale = size / min(width, height)
    resized = image.resize(
        (max(size, round(width * scale)), max(size, round(height * scale))), Image.Resampling.BICUBIC
    )
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    return np.asarray(resized.crop((left, top, left + size, top + size)), dtype=np.uint8)


def read_archive(path: Path, size: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {"train": (images, labels), "val": (images, labels)} from the tgz."""
    index = {wnid: i for i, wnid in enumerate(CLASSES)}
    images: dict[str, list[np.ndarray]] = {"train": [], "val": []}
    labels: dict[str, list[int]] = {"train": [], "val": []}
    with tarfile.open(path) as archive:
        for member in archive:
            parts = Path(member.name).parts
            # Expected layout is imagenette2-160/<split>/<wnid>/<file>.JPEG
            if not member.isfile() or len(parts) != 4 or parts[0] != ARCHIVE_ROOT:
                continue
            split, wnid = parts[1], parts[2]
            if split not in images or wnid not in index:
                raise ValueError(f"unexpected archive member {member.name}")
            images[split].append(resize_and_crop(archive.extractfile(member).read(), size))
            labels[split].append(index[wnid])
            count = sum(len(v) for v in images.values())
            if count % 2000 == 0:
                print(f"resize: {count} images", flush=True)
    return {
        split: (np.stack(images[split]), np.asarray(labels[split], dtype=np.int64))
        for split in images
    }


def split_per_class(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into two halves with the same class proportions."""
    rng = np.random.default_rng(seed)
    first, second = [], []
    for label in np.unique(labels):
        members = rng.permutation(np.flatnonzero(labels == label))
        half = (len(members) + 1) // 2
        first.append(members[:half])
        second.append(members[half:])
    return np.sort(np.concatenate(first)), np.sort(np.concatenate(second))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="data/imagenette_128.npz")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--archive", help="use an already downloaded imagenette2-160.tgz")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        print(f"{output} already exists; delete it to rebuild")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="imagenette-") as directory:
        if args.archive:
            archive = Path(args.archive)
            source_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
        else:
            archive = Path(directory) / "imagenette2-160.tgz"
            print(f"downloading {URL}", flush=True)
            source_sha256 = download(URL, archive)
        splits = read_archive(archive, args.size)

        train_images, train_labels = splits["train"]
        val_images, val_labels = splits["val"]
        val_index, test_index = split_per_class(val_labels, SPLIT_SEED)
        arrays = {
            "train_images": train_images, "train_labels": train_labels,
            "val_images": val_images[val_index], "val_labels": val_labels[val_index],
            "test_images": val_images[test_index], "test_labels": val_labels[test_index],
            "class_names": np.asarray(list(CLASSES.values())),
            "wnids": np.asarray(list(CLASSES)),
            "source_url": np.asarray(URL),
            "source_sha256": np.asarray(source_sha256),
            "split_seed": np.asarray(SPLIT_SEED),
        }
        # Write next to the target and rename, so an interrupted run leaves no partial file
        partial = Path(directory) / output.name
        with partial.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        shutil.move(str(partial), output)

    for split in ("train", "val", "test"):
        counts = np.bincount(arrays[f"{split}_labels"], minlength=len(CLASSES))
        print(f"{split}: {len(arrays[f'{split}_labels'])} images, per class {counts.tolist()}")
    print(f"wrote {output} ({output.stat().st_size / 1e6:.0f} MB), source sha256 {source_sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
