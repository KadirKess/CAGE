"""Unified Trainer for all three CAGE training modes: cnn, unet_mse, cage."""

import math
import random
from pathlib import Path
from time import perf_counter

import torch
import torch.optim as optim
from cage.artifacts import append_epoch_log, save_accuracy_curves, save_loss_curves
from cage.data import NUM_CLASSES, add_gaussian_noise, get_dataloaders
from cage.losses import cage_loss, ce_loss, ms_ssim_loss, mse_loss
from cage.models.cnn import WeakCNN
from cage.models.unet import UNet
from cage.repro import configure_tf32

AMP_DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
# Plain status lines instead of progress bars, which read the same in a
# terminal, a Colab cell and a --jobs log. Long phases report this often.
PROGRESS_SECONDS = 30


class Trainer:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.mode = config["mode"]
        self.device = torch.device(config["device"])
        self.run_name = config["run_name"]
        self.epochs = config["epochs"]
        self.lr = config["lr"]
        self.compile_models = config.get("compile", False)
        self.compile_backend = config.get("compile_backend", "inductor")
        self.precision = config.get("precision", "fp32")
        if self.precision not in AMP_DTYPES:
            raise ValueError("precision must be fp32, fp16 or bf16")
        self._amp_dtype = AMP_DTYPES[self.precision]
        self._amp_enabled = self._amp_dtype is not None
        self.tf32 = config.get("tf32", False)
        configure_tf32(self.tf32)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        # bf16 has the FP32 exponent range, so only fp16 needs loss scaling
        self.scaler = torch.amp.GradScaler(
            self.device.type, init_scale=config.get("amp_init_scale", 1024.0),
            enabled=self.precision == "fp16",
        )
        self._skipped_steps = 0
        self._epoch = 0
        self._last_progress = 0.0
        self.sigmas = config["noise_sigmas"]
        self.lambda1 = config.get("lambda1", 0.5)
        self.lambda2 = config.get("lambda2", 0.5)
        self.rec_loss = config.get("rec_loss", "mse")
        self._rec_fn = ms_ssim_loss if self.rec_loss == "msssim" else mse_loss
        self.metrics_dir = config.get("metrics_dir", "outputs/metrics")
        self.checkpoints_dir = config.get("checkpoints_dir", "outputs/checkpoints")
        self.plots_dir = config.get("plots_dir", "outputs/plots")
        self._cnn_checkpoint = config.get(
            "cnn_checkpoint",
            str(Path(self.checkpoints_dir) / "cnn.pt"),
        )

        self.loaders = get_dataloaders(
            data_dir=config.get("data_dir", "data"),
            batch_size=config.get("batch_size", 128),
            num_workers=config.get("num_workers", 0),
        )
        self._setup_models()
        if self.compile_models:
            self._log(f"enabling torch.compile backend={self.compile_backend}; "
                      "the first training and validation batches take longer")
            self.cnn.compile(backend=self.compile_backend)
            if self.mode != "cnn":
                self.unet.compile(backend=self.compile_backend)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=config.get("lr_scheduler_patience", 10),
            factor=config.get("lr_scheduler_factor", 0.5), threshold=0.0, threshold_mode="abs",
        )

    def _setup_models(self) -> None:
        num_classes = self.config.get("num_classes", NUM_CLASSES)
        if self.mode == "cnn":
            self.cnn = WeakCNN(num_classes=num_classes).to(self.device)
            self.optimizer = optim.Adam(self.cnn.parameters(), lr=self.lr)
        else:
            self.unet = UNet().to(self.device)
            self.cnn = self._load_frozen_cnn()
            self.optimizer = optim.Adam(self.unet.parameters(), lr=self.lr)

    def _load_frozen_cnn(self) -> WeakCNN:
        num_classes = self.config.get("num_classes", NUM_CLASSES)
        cnn = WeakCNN(num_classes=num_classes).to(self.device)
        cnn.load_state_dict(torch.load(self._cnn_checkpoint, map_location=self.device))
        cnn.freeze()
        return cnn

    def train(self) -> list[dict]:
        patience: int = self.config.get("early_stopping_patience", 20)
        min_delta: float = self.config.get(
            "early_stopping_min_delta",
            self.config.get("early_stopping_threshold", 0.0),
        )
        best_val_loss: float = float("inf")
        stopping_best_loss: float = float("inf")
        best_epoch = 0
        patience_counter: int = 0
        logs: list[dict] = []

        self._log(f"device={self.device}, precision={self.precision}, tf32={self.tf32}, "
                  f"gradient_scaling={self.scaler.is_enabled()}, max_grad_norm={self.max_grad_norm}, "
                  f"compile={self.compile_models}, {self.epochs} epochs max")
        for epoch in range(1, self.epochs + 1):
            self._epoch = epoch
            epoch_started = perf_counter()
            train_loss, train_rec, train_ce = self._train_epoch()
            val_loss, val_rec, val_ce, acc_per_sigma = self._val_epoch()

            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise FloatingPointError(f"{self.run_name}: non-finite epoch loss; checkpoint not updated")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                self._save_checkpoint()
            if (stopping_best_loss - val_loss) > min_delta:
                stopping_best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            previous_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr != previous_lr:
                self._log(f"validation plateau; lr {previous_lr:g} -> {current_lr:g}")

            entry: dict = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "best_val_loss": round(best_val_loss, 6),
                "best_epoch": best_epoch,
                "precision": self.precision,
                "tf32": self.tf32,
                "amp_skipped_steps": self._skipped_steps,
                "grad_scale": self.scaler.get_scale(),
                "learning_rate": current_lr,
                "epoch_seconds": round(perf_counter() - epoch_started, 3),
                "compile": self.compile_models,
                "compile_backend": self.compile_backend if self.compile_models else None,
                "mode": self.mode,
                "rec_loss": self.rec_loss,
                "lambda1": self.lambda1,
                "lambda2": self.lambda2,
            }
            if train_rec is not None:
                entry["train_rec_loss"] = round(train_rec, 6)
                entry["train_ce_loss"] = round(train_ce, 6)
                entry["val_rec_loss"] = round(val_rec, 6)
                entry["val_ce_loss"] = round(val_ce, 6)
            if acc_per_sigma:
                entry["cnn_accuracy_per_sigma"] = {
                    str(s): round(a, 4) for s, a in acc_per_sigma.items()
                }

            append_epoch_log(self.run_name, entry, metrics_dir=self.metrics_dir)
            logs.append(entry)
            accuracy = "".join(f" acc@{s}={a:.3f}" for s, a in acc_per_sigma.items())
            self._log(
                f"epoch {epoch}/{self.epochs} train={train_loss:.4f} val={val_loss:.4f} "
                f"best={best_val_loss:.4f}@{best_epoch} lr={current_lr:g}{accuracy} "
                f"patience={patience_counter}/{patience} {entry['epoch_seconds']:.1f}s"
            )

            if patience_counter >= patience:
                self._log(f"early stopping at epoch {epoch} "
                          f"(no improvement > {min_delta} for {patience} epochs)")
                break

        save_loss_curves(self.run_name, logs, plots_dir=self.plots_dir)
        if any("cnn_accuracy_per_sigma" in e for e in logs):
            save_accuracy_curves(self.run_name, logs, plots_dir=self.plots_dir)

        self._log(f"finished at epoch {logs[-1]['epoch']}, best val={best_val_loss:.4f} at epoch {best_epoch}")
        return logs

    def _log(self, message: str) -> None:
        print(f"{self.run_name}: {message}", flush=True)

    def _batch_progress(self, phase: str, done: int, total: int, started: float) -> None:
        now = perf_counter()
        if now - self._last_progress >= PROGRESS_SECONDS:
            self._last_progress = now
            self._log(f"epoch {self._epoch}/{self.epochs} {phase}: {done}/{total} batches, "
                      f"{now - started:.0f}s elapsed")

    def _train_epoch(self) -> tuple[float, float | None, float | None]:
        (self.cnn if self.mode == "cnn" else self.unet).train()
        total_loss, total_rec, total_ce, n = 0.0, 0.0, 0.0, 0
        self._skipped_steps = 0
        successful_steps = 0
        started = self._last_progress = perf_counter()

        for step, (x, y) in enumerate(self.loaders["train"], start=1):
            x, y = x.to(self.device), y.to(self.device).long().view(-1)
            self.optimizer.zero_grad()

            loss, l_rec, l_ce = self._batch_loss(x, y)
            if l_rec is not None:
                total_rec += l_rec.item() * x.size(0)
                total_ce += l_ce.item() * x.size(0)

            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"{self.run_name}: non-finite training loss")
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            model = self.cnn if self.mode == "cnn" else self.unet
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            if not torch.isfinite(grad_norm).item():
                if not self.scaler.is_enabled():
                    raise FloatingPointError(f"{self.run_name}: non-finite gradients")
                self._skipped_steps += 1
                self.scaler.update(new_scale=self.scaler.get_scale() * 0.5)
                self._log(f"skipped non-finite gradients; scale={self.scaler.get_scale():g}")
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                successful_steps += 1
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            self._batch_progress("training", step, len(self.loaders["train"]), started)

        if successful_steps == 0:
            raise FloatingPointError(f"{self.run_name}: every optimizer step overflowed; stop and inspect FP16")
        if self.mode == "cage":
            return total_loss / n, total_rec / n, total_ce / n
        return total_loss / n, None, None

    def _val_epoch(self) -> tuple[float, float | None, float | None, dict]:
        (self.cnn if self.mode == "cnn" else self.unet).eval()
        total_loss, total_rec, total_ce, n = 0.0, 0.0, 0.0, 0
        acc_per_sigma: dict = {}

        with torch.no_grad():
            for x, y in self.loaders["val"]:
                x, y = x.to(self.device), y.to(self.device).long().view(-1)
                loss, l_rec, l_ce = self._batch_loss(x, y)
                if l_rec is not None:
                    total_rec += l_rec.item() * x.size(0)
                    total_ce += l_ce.item() * x.size(0)
                total_loss += loss.item() * x.size(0)
                n += x.size(0)

            if self.mode != "cnn":
                for sigma in self.sigmas:
                    correct, total = 0, 0
                    for x, y in self.loaders["val"]:
                        x, y = x.to(self.device), y.to(self.device).long().view(-1)
                        with self._autocast():
                            x_denoised = self.unet(add_gaussian_noise(x, sigma))
                            logits = self.cnn(x_denoised)
                        if not torch.isfinite(logits).all().item():
                            raise FloatingPointError(f"{self.run_name}: non-finite validation logits")
                        preds = logits.argmax(dim=1)
                        correct += (preds == y).sum().item()
                        total += y.size(0)
                    acc_per_sigma[sigma] = correct / total

        if self.mode == "cage":
            return total_loss / n, total_rec / n, total_ce / n, acc_per_sigma
        return total_loss / n, None, None, acc_per_sigma

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type, dtype=self._amp_dtype or torch.float32, enabled=self._amp_enabled
        )

    def _batch_loss(self, x, y):
        with self._autocast():
            if self.mode == "cnn":
                return ce_loss(self.cnn(x), y), None, None
            noisy = add_gaussian_noise(x, random.choice(self.sigmas))
            denoised = self.unet(noisy)
            if self.mode == "unet_mse":
                return self._rec_fn(denoised, x), None, None
            return cage_loss(denoised, x, self.cnn(denoised), y,
                             self.lambda1, self.lambda2, rec_loss_fn=self._rec_fn)

    def _save_checkpoint(self) -> None:
        Path(self.checkpoints_dir).mkdir(parents=True, exist_ok=True)
        model = self.cnn if self.mode == "cnn" else self.unet
        torch.save(
            model.state_dict(),
            Path(self.checkpoints_dir) / f"{self.run_name}.pt",
        )
