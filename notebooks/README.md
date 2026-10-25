# Notebooks

`CAGE.ipynb` runs the v0.1 study on Google Colab. It mounts Drive, clones the
repository to `/content/CAGE`, builds the locked `.venv`, prepares or copies
the Imagenette npz, and then calls `cage.run` as shell commands for training,
evaluation, aggregation and analysis. The last cells plot training curves, the
aggregate table and the release figures with plain Python from the files on
Drive. See the Colab section of `docs/reproducing.md`.

Locally, use the command-line runner directly; it is the canonical interface
and the only supported path for results committed to `results/v0.1/`. The
notebook is committed without outputs.
