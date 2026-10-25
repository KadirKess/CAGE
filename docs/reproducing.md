# Reproducing the study

## Setup

```bash
uv sync --locked --extra dev
uv run --locked --extra dev python scripts/check_environment.py
uv run --locked --extra dev pytest -v
```

uv uses the project-local `.venv` and Python 3.10 from `.python-version`. `--locked` rejects an outdated `uv.lock` instead of changing it. Include `--extra dev` when running tests so pytest stays installed. `requirements.txt` records versions only.

On Apple Silicon, uv selects the macOS 12 wheel of SciPy 1.15.3, because the macOS 14 wheel fails to import PROPACK on the M1 Pro.

`check_environment.py` logs interpreter and package versions, import timings and accelerator availability. Pass `--require-mps` on a Mac. Sandboxed processes may report MPS as unavailable, so run it from your own terminal.

## Data

```bash
uv run --locked python scripts/prepare_imagenette.py
```

This downloads the official 160px Imagenette release (about 100 MB), converts each image to RGB, resizes the shorter side to 128 (bicubic) and center crops to 128x128. The official train split (9,469 images) is the training set. The official validation split (3,925 images) is divided per class with a fixed seed into validation (1,966) and test (1,959). The output is one npz of about 510 MB holding uint8 images, labels, class names, the archive's SHA-256 and the split seed.

## Running the matrix

`experiments/configs/v0.1.json` expands to 21 configurations x 3 seeds (17, 43, 71).

```bash
uv run --locked python -m cage.run train --config experiments/configs/v0.1.json --dry-run
uv run --locked python -m cage.run train --config experiments/configs/v0.1.json
uv run --locked python -m cage.run evaluate --study v0.1
uv run --locked python -m cage.run aggregate --study v0.1
uv run --locked python -m cage.run analyze --study v0.1
```

`--run-id <id>` trains one configuration. `--jobs N` trains N runs at once, each in its own process, so results match serial training. WeakCNN baselines run first, and a denoiser starts once its seed's baseline is complete.

For local development, `experiments/configs/v0.1-smoke.json` is the same matrix with one seed, two epochs, FP32 and no compilation, under study name `v0.1-smoke`:

```bash
uv run --locked python -m cage.run train --config experiments/configs/v0.1-smoke.json --jobs 2
```

## Outputs

Each run writes to `runs/<study>/runs/<config id>/seed-<seed>/`:

- `checkpoints/<run_name>.pt`, the lowest-validation-loss checkpoint
- `metrics/<run_name>.json`, the per-epoch log
- `plots/<run_name>_*.png`, loss and accuracy curves
- `manifest.json`, the resolved config, seed state, package versions, git commit and checkpoint hash
- `train.log`, console output when trained with `--jobs`

The manifest is written last, so it marks a run as complete. A run with a checkpoint but no manifest was interrupted, and the runner retrains it from scratch. Runs never resume mid-training.

`evaluate` writes `runs/<study>/evaluations/seed-<seed>.json`. `aggregate` writes `results/<study>/per_seed_metrics.csv` and `summary.json`, and `analyze` writes `results/<study>/figures/` and `gradcam.json`. The GradCAM analysis and the qualitative figure need the checkpoints under `runs/` and the dataset. Without them, `analyze` skips both. `runs/` is ignored by git; `results/` is meant to be committed.

## Training details

- Adam, lr 1e-3, batch 64, at most 500 epochs.
- ReduceLROnPlateau on validation loss: patience 10, factor 0.1, zero threshold.
- Early stopping with patience 20, reset by any strictly lower validation loss. Every lower finite validation loss saves a checkpoint.
- Gradients clipped to norm 1.0. A non-finite loss or gradient aborts training before saving.
- Noise is sampled fresh per batch. Multi-sigma runs draw one sigma per batch.
- Training is seeded but uses non-deterministic GPU kernels. Test corruption is regenerated from the sample index, sigma and one evaluation seed (20260715), so every model sees the same noisy test images.
- Test evaluation is eager FP32 with TF32 off.
- Intervals are two-sided 95% Student-t across seeds. Comparisons against controls are paired within seed.
- Each split lives in memory as uint8 and batches are gathered in one operation, so `num_workers: 0` is enough (about 1.3 ms per batch of 64 on the M1 Pro).

## Precision and hardware

The campaign ran on a Colab L4 with BF16 autocast, `torch.compile` (Inductor) and TF32 off. Parameters, Adam state and the losses stay FP32. BF16 has the FP32 exponent range, so there is no gradient scaling. The trainer also supports `precision: fp32`, `precision: fp16` (dynamic loss scaling, skipped updates on overflow) and `tf32: true`.

`scripts/colab_benchmark.py` measured a CAGE step on the L4 (batch 64, torch 2.11, with the earlier single-channel models):

| Setting | MSE term | MS-SSIM term |
|---|---|---|
| FP32 | 82.5 ms | 91.4 ms |
| TF32 | 78.4 ms | |
| BF16 | 46.3 ms | |
| BF16 + compile | 34.8 ms | 44.2 ms |

FP16 with compilation matched BF16 but needs loss scaling. After 50 seeded steps every setting's loss was within 0.3% of FP32. The RGB models only change the first and last convolutions, but the benchmark should be rerun to confirm the ranking.

One job saturates the L4: 2 to 6 concurrent jobs gave about 8% less total throughput, so Colab uses `--jobs 1`. On the M1 Pro, two jobs gave 1.7x the throughput of one, and more gave nothing.

On Apple MPS a step took 211 ms in FP32 and 362 ms under FP16 autocast, and Inductor could not build Metal kernels. The Mac can't run `v0.1.json` as configured; use the smoke config.

## Colab

`notebooks/CAGE.ipynb` runs the study on a GPU runtime (L4). The first cell mounts Drive, clones or pulls the repo into `/content/CAGE` and builds `.venv` with `uv sync --locked`. Later cells call `cage.run` as shell commands and plot the results. Colab clones from GitHub, so push code changes first.

Drive keeps `datasets/imagenette_128.npz`, `runs/` and `results/` under `MyDrive/CAGE/`. After a disconnect, rerun the training cell: completed runs are skipped and interrupted ones restart.
