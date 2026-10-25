"""Standalone CAGE training-step benchmark for a Colab GPU runtime.

Paste the whole file into one Colab cell (or run `python colab_benchmark.py`).
It needs no clone of the repository: the U-Net, WeakCNN and losses below are
copies of cage/models and cage/losses, so keep them in sync if those change.

What it measures, on random 128x128 RGB batches shaped like Imagenette:
  1. ms per CAGE training step for every precision / TF32 / compile setting,
     for both reconstruction losses (MSE and MS-SSIM), plus the loss after a
     fixed number of seeded steps as a rough numerics sanity check.
  2. DataLoader cost per batch for cage/data.py's in-memory path (one uint8
     gather and conversion per batch), so you can see whether the CPU keeps up.
  3. Total throughput with K training processes sharing the GPU, which is what
     `cage.run train --jobs K` does.

Every measurement runs in a fresh subprocess, so a crashing torch.compile or a
leaked CUDA context cannot affect the next one.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Settings. Adjust and rerun.
BATCH_SIZE = 64
WARMUP_STEPS = 10  # compile settings get 3x this, compilation happens here
TIMED_STEPS = 50
PRECISIONS = ["fp32", "fp32_tf32", "fp16", "bf16"]
COMPILE = [False, True]
REC_LOSSES = ["mse", "msssim"]
LOADER_WORKERS = [0, 2]
LOADER_BATCHES = 100
# Concurrency runs use the settings below. Change them to the winner of part 1.
CONCURRENCY = [1, 2, 3, 4, 6]
CONCURRENCY_SETTING = {"precision": "bf16", "compile": True, "rec_loss": "mse"}
CONCURRENCY_SECONDS = 30

WORKER_SOURCE = r'''
import json, os, random, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim as _ms_ssim_fn


class _DoubleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = _DoubleConvBlock(channels, 16)
        self.enc2 = _DoubleConvBlock(16, 32)
        self.enc3 = _DoubleConvBlock(32, 64)
        self.bottleneck = _DoubleConvBlock(64, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = _DoubleConvBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = _DoubleConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = _DoubleConvBlock(32, 16)
        self.out = nn.Sequential(nn.Conv2d(16, channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class WeakCNN(nn.Module):
    def __init__(self, num_classes=10, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(16 * 30 * 30, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()


def mse_loss(x_pred, x_true):
    with torch.autocast(device_type=x_pred.device.type, enabled=False):
        return F.mse_loss(x_pred.float(), x_true.float())


def ms_ssim_loss(x_pred, x_true):
    with torch.autocast(device_type=x_pred.device.type, enabled=False):
        return 1.0 - _ms_ssim_fn(x_pred.float(), x_true.float(), data_range=1.0,
                                 size_average=True, win_size=7)


def ce_loss(logits, labels):
    with torch.autocast(device_type=logits.device.type, enabled=False):
        return F.cross_entropy(logits.float(), labels)


def set_tf32(enabled):
    # torch >= 2.9 has the fp32_precision API and refuses a mix with allow_tf32
    mode = "tf32" if enabled else "ieee"
    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        torch.backends.cudnn.conv.fp32_precision = mode
        torch.backends.cuda.matmul.fp32_precision = mode
    else:
        torch.backends.cudnn.allow_tf32 = enabled
        torch.backends.cuda.matmul.allow_tf32 = enabled


def step_benchmark(args):
    device = torch.device("cuda")
    precision = args["precision"]
    set_tf32(precision == "fp32_tf32")
    torch.manual_seed(0)
    random.seed(0)
    unet = UNet().to(device)
    cnn = WeakCNN().to(device)
    cnn.freeze()
    if args["compile"]:
        unet.compile()
        cnn.compile()
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=precision == "fp16")
    rec_fn = ms_ssim_loss if args["rec_loss"] == "msssim" else mse_loss

    # A fixed pool of batches so every setting sees identical data
    generator = torch.Generator(device="cpu").manual_seed(0)
    pool = [(torch.rand(args["batch_size"], 3, 128, 128, generator=generator),
             torch.randint(0, 10, (args["batch_size"],), generator=generator)) for _ in range(8)]
    pool = [(x.to(device), y.to(device)) for x, y in pool]
    skipped = 0

    def step(i):
        nonlocal skipped
        x, y = pool[i % len(pool)]
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            noisy = (x + torch.randn_like(x) * 0.3).clamp(0.0, 1.0)
            denoised = unet(noisy)
            l_rec = rec_fn(denoised, x)
            l_ce = ce_loss(cnn(denoised), y)
            loss = 0.5 * l_rec + 0.5 * l_ce
        # The trainer syncs on these every step, so the benchmark does too
        l_rec.item(); l_ce.item()
        if not torch.isfinite(loss).item():
            raise FloatingPointError("non-finite loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        if not torch.isfinite(grad_norm).item():
            skipped += 1
            scaler.update(new_scale=scaler.get_scale() * 0.5)
        else:
            scaler.step(optimizer)
            scaler.update()
        return loss.item()

    warmup = args["warmup"] * (3 if args["compile"] else 1)
    compile_started = time.perf_counter()
    for i in range(warmup):
        step(i)
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - compile_started

    if args.get("barrier_dir"):
        # In concurrency mode, wait for every process to finish warmup, then run for a fixed time
        barrier = args["barrier_dir"]
        open(os.path.join(barrier, f"ready-{os.getpid()}"), "w").close()
        while not os.path.exists(os.path.join(barrier, "go")):
            time.sleep(0.05)
        started = time.perf_counter()
        steps = 0
        while time.perf_counter() - started < args["seconds"]:
            step(warmup + steps)
            steps += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return {"steps": steps, "seconds": elapsed,
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss = None
    for i in range(args["steps"]):
        loss = step(warmup + i)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "ms_per_step": elapsed / args["steps"] * 1000,
        "images_per_s": args["steps"] * args["batch_size"] / elapsed,
        "warmup_s": warmup_seconds,
        "final_loss": loss,
        "skipped_steps": skipped,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def loader_benchmark(args):
    from torch.utils.data import DataLoader, Dataset

    class InMemoryImagenette(Dataset):
        # Same path as cage/data.py, uint8 CHW in memory and one gather per batch
        def __init__(self, n):
            generator = torch.Generator().manual_seed(0)
            self.images = torch.randint(0, 256, (n, 3, 128, 128), dtype=torch.uint8, generator=generator)
            self.labels = torch.randint(0, 10, (n,), generator=generator)

        def __len__(self):
            return len(self.labels)

        def __getitems__(self, indices):
            index = torch.as_tensor(indices)
            return self.images[index].float().div_(255.0), self.labels[index]

    workers = args["workers"]
    options = {"num_workers": workers, "collate_fn": lambda batch: batch}
    if workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(InMemoryImagenette(9469), batch_size=args["batch_size"], shuffle=True, **options)
    iterator = iter(loader)
    for _ in range(10):
        next(iterator)
    started = time.perf_counter()
    for _ in range(args["batches"]):
        x, _ = next(iterator)
        x.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    return {"ms_per_batch": (time.perf_counter() - started) / args["batches"] * 1000}


if __name__ == "__main__":
    args = json.loads(sys.argv[1])
    try:
        result = loader_benchmark(args) if args["kind"] == "loader" else step_benchmark(args)
        result["ok"] = True
    except Exception as error:
        result = {"ok": False, "error": f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"}
    print("RESULT " + json.dumps(result), flush=True)
'''


def ensure_dependencies():
    try:
        import pytorch_msssim  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch-msssim==1.0.0"], check=True)


def write_worker():
    path = Path(tempfile.gettempdir()) / "cage_bench_worker.py"
    path.write_text(WORKER_SOURCE)
    return path


def launch(worker, args):
    return subprocess.Popen(
        [sys.executable, str(worker), json.dumps(args)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def collect(process, timeout=1800):
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        return {"ok": False, "error": "timed out"}
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    tail = (stderr.strip().splitlines() or ["no output"])[-1]
    return {"ok": False, "error": tail[:200]}


def run_one(worker, args):
    return collect(launch(worker, args))


def describe_machine():
    import torch

    print(f"python {sys.version.split()[0]}, torch {torch.__version__}, cuda {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU. In Colab: Runtime > Change runtime type > L4 GPU.")
    props = torch.cuda.get_device_properties(0)
    print(f"gpu {props.name}, {props.total_memory / 1e9:.1f} GB, bf16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"cpu cores visible: {os.cpu_count()}")
    print()


def step_section(worker):
    print(f"Part 1: CAGE training step, batch {BATCH_SIZE}, {TIMED_STEPS} timed steps")
    header = f"{'rec':7s} {'precision':10s} {'compile':7s} {'ms/step':>8s} {'img/s':>7s} " \
             f"{'warmup s':>8s} {'loss':>8s} {'skipped':>7s} {'mem GB':>6s}"
    print(header)
    results = []
    for rec_loss in REC_LOSSES:
        baseline = None
        for precision in PRECISIONS:
            for compile_ in COMPILE:
                args = {"kind": "step", "precision": precision, "compile": compile_, "rec_loss": rec_loss,
                        "batch_size": BATCH_SIZE, "warmup": WARMUP_STEPS, "steps": TIMED_STEPS}
                result = run_one(worker, args)
                results.append({**args, **result})
                label = f"{rec_loss:7s} {precision:10s} {str(compile_):7s}"
                if not result["ok"]:
                    print(f"{label} FAILED {result['error']}")
                    continue
                if baseline is None and precision == "fp32" and not compile_:
                    baseline = result["ms_per_step"]
                speedup = f"  {baseline / result['ms_per_step']:.2f}x vs fp32" if baseline else ""
                print(f"{label} {result['ms_per_step']:8.1f} {result['images_per_s']:7.0f} "
                      f"{result['warmup_s']:8.1f} {result['final_loss']:8.4f} {result['skipped_steps']:7d} "
                      f"{result['peak_mem_gb']:6.2f}{speedup}", flush=True)
    print("Losses after the same seeded steps should be close across precisions; a large gap or")
    print("skipped steps means that precision changes training, not just speed.")
    print()
    return results


def loader_section(worker):
    print(f"Part 2: DataLoader, in-memory batched gather, batch {BATCH_SIZE}")
    results = []
    for workers in LOADER_WORKERS:
        args = {"kind": "loader", "workers": workers, "batch_size": BATCH_SIZE, "batches": LOADER_BATCHES}
        result = run_one(worker, args)
        results.append({**args, **result})
        if result["ok"]:
            print(f"num_workers={workers}: {result['ms_per_batch']:.1f} ms/batch", flush=True)
        else:
            print(f"num_workers={workers}: FAILED {result['error']}")
    print("If this is close to the step time, the GPU waits on the CPU, and parallel jobs share these cores.")
    print()
    return results


def concurrency_section(worker):
    setting = CONCURRENCY_SETTING
    print(f"Part 3: K processes sharing the GPU, {setting}, {CONCURRENCY_SECONDS}s each")
    results = []
    single = None
    for k in CONCURRENCY:
        barrier = tempfile.mkdtemp(prefix="cage-bench-")
        args = {"kind": "step", **setting, "batch_size": BATCH_SIZE, "warmup": WARMUP_STEPS,
                "barrier_dir": barrier, "seconds": CONCURRENCY_SECONDS}
        processes = [launch(worker, args) for _ in range(k)]
        deadline = time.time() + 900
        while len([p for p in os.listdir(barrier) if p.startswith("ready-")]) < k:
            if time.time() > deadline or any(p.poll() is not None for p in processes):
                break
            time.sleep(0.2)
        Path(barrier, "go").touch()
        outcomes = [collect(p) for p in processes]
        failed = [o for o in outcomes if not o["ok"]]
        if failed:
            print(f"K={k}: FAILED {failed[0]['error']}")
            results.append({"k": k, "ok": False})
            break
        throughput = sum(o["steps"] * BATCH_SIZE / o["seconds"] for o in outcomes)
        per_job_ms = sum(o["seconds"] / o["steps"] * 1000 for o in outcomes) / k
        single = single or throughput
        memory = max(o["peak_mem_gb"] for o in outcomes)
        print(f"K={k}: {throughput:7.0f} img/s total, {throughput / single:.2f}x vs K=1, "
              f"{per_job_ms:.1f} ms/step per job, peak {memory:.2f} GB per job", flush=True)
        results.append({"k": k, "ok": True, "images_per_s": throughput, "ms_per_step": per_job_ms})
    print("Pick the smallest K past which total img/s stops growing; that is --jobs.")
    print()
    return results


def main():
    ensure_dependencies()
    describe_machine()
    worker = write_worker()
    started = time.time()
    report = {
        "steps": step_section(worker),
        "loader": loader_section(worker),
        "concurrency": concurrency_section(worker),
    }
    out = Path("cage_benchmark.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"Done in {time.time() - started:.0f}s; raw numbers in {out.resolve()}")


main()
