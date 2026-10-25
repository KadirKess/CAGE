import os

# CAGE only saves figures to files. Force the non-interactive backend before
# anything imports matplotlib. A Colab kernel exports an inline MPLBACKEND that
# `!` subprocesses inherit, and the locked .venv has no matplotlib_inline, so
# matplotlib would fail at import.
os.environ["MPLBACKEND"] = "Agg"
