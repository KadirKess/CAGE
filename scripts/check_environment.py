"""Check the local research environment without loading data or training."""

import argparse
import importlib
import importlib.metadata
import logging
from pathlib import Path
import platform
import sys
from time import perf_counter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-mps", action="store_true", help="fail if MPS is unavailable")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.info("Python %s (%s)", platform.python_version(), sys.executable)
    logging.info("Platform: %s", platform.platform())
    failures = []
    if sys.version_info[:2] != (3, 10):
        failures.append("Python 3.10 is required; run uv sync --locked --extra dev")
    modules = (
        ("torch", "torch"), ("torchvision", "torchvision"),
        ("pillow", "PIL"), ("numpy", "numpy"),
        ("matplotlib", "matplotlib"), ("scipy", "scipy.signal"),
        ("scikit-learn", "sklearn"), ("torchmetrics", "torchmetrics.image"),
        ("grad-cam", "pytorch_grad_cam"), ("pytorch-msssim", "pytorch_msssim"),
        ("pytest", "pytest"),
    )
    for package, module in modules:
        started = perf_counter()
        try:
            importlib.import_module(module)
            logging.info("%s %s imported in %.2fs", package, importlib.metadata.version(package), perf_counter() - started)
        except Exception as error:
            failures.append(f"{package}: {type(error).__name__}: {error}")
    try:
        import torch

        logging.info("MPS built=%s available=%s; CUDA available=%s",
                     torch.backends.mps.is_built(), torch.backends.mps.is_available(), torch.cuda.is_available())
        if args.require_mps and not torch.backends.mps.is_available():
            failures.append("MPS is unavailable in this process")
        value = torch.ones(4, requires_grad=True)
        value.square().sum().backward()
        logging.info("CPU tensor forward/backward passed")
    except Exception as error:
        failures.append(f"PyTorch smoke check: {type(error).__name__}: {error}")
    logging.info("Environment: %s", Path(sys.prefix))
    for failure in failures:
        logging.error("%s", failure)
    if failures:
        logging.error("Environment check failed with %d issue(s)", len(failures))
        return 1
    logging.info("Environment check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
