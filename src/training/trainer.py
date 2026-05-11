"""Training loop for segmentation models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.evaluation.metrics import dice_per_class, mean_dice, pixel_accuracy
from src.utils.logging import setup_logger

logger = setup_logger("trainer")


class Trainer:
    """Generic segmentation trainer.

    Args:
        model:        PyTorch model.
        criterion:    Loss function (logits, targets) → scalar.
        optimizer:    Torch optimizer.
        scheduler:    Optional LR scheduler (step called after each epoch).
        device:       "cuda" | "cpu" | "mps".
        num_classes:  Number of output classes.
        checkpoint_dir: Directory to save best model weights.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        device: str = "cuda",
        num_classes: int = 4,
        checkpoint_dir: str | Path = "checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_dice = -1.0
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        self.model.train(train)
        total_loss = 0.0
        total_dice = 0.0
        total_acc  = 0.0
        n_batches  = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for images, masks in loader:
                images = images.to(self.device, non_blocking=True)
                masks  = masks.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss   = self.criterion(logits, masks)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                preds = logits.argmax(dim=1)
                dice  = mean_dice(preds.cpu(), masks.cpu(), self.num_classes)
                acc   = pixel_accuracy(preds.cpu(), masks.cpu())

                total_loss += loss.item()
                total_dice += dice
                total_acc  += acc
                n_batches  += 1

        return {
            "loss": total_loss / n_batches,
            "dice": total_dice / n_batches,
            "acc":  total_acc  / n_batches,
        }

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        experiment_name: str = "model",
    ) -> list[dict[str, Any]]:
        """Run the full training loop.

        Returns:
            Training history (list of per-epoch metric dicts).
        """
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics   = self._run_epoch(val_loader,   train=False)

            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - t0
            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "train_acc":  train_metrics["acc"],
                "val_loss":   val_metrics["loss"],
                "val_dice":   val_metrics["dice"],
                "val_acc":    val_metrics["acc"],
                "lr":         self.optimizer.param_groups[0]["lr"],
                "elapsed_s":  elapsed,
            }
            self.history.append(record)

            logger.info(
                f"Epoch {epoch:03d}/{epochs}  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"val_dice={val_metrics['dice']:.4f}  "
                f"lr={record['lr']:.2e}  "
                f"({elapsed:.0f}s)"
            )

            # Save best checkpoint
            if val_metrics["dice"] > self.best_val_dice:
                self.best_val_dice = val_metrics["dice"]
                ckpt_path = self.checkpoint_dir / f"{experiment_name}_best.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_dice": self.best_val_dice,
                    },
                    ckpt_path,
                )
                logger.info(f"  ✓ New best val_dice={self.best_val_dice:.4f} — saved to {ckpt_path}")

        return self.history
