# Contributing

Please open an issue before proposing a change that alters the experimental
protocol, architecture, dataset, loss, or reported finding. The README reports
the study, so protocol changes must update it in the same pull request.

Use Python 3.10 and install the locked environment with
`uv sync --locked --extra dev`. Run `uv run --locked --extra dev pytest -v`
before opening a pull request. Tests must use synthetic data and
must not download Imagenette or require an accelerator.

Do not commit datasets, checkpoints, or files below `runs/`. Commit only
reproducible code, configurations, and approved derived artifacts in
`results/`.
