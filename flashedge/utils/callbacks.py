"""Training callbacks for checkpointing and early stopping."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        mode: "min" for loss, "max" for accuracy.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """Check if training should stop.

        Args:
            value: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


class ModelCheckpoint:
    """Save model checkpoints during training.

    Args:
        save_dir: Directory to save checkpoints.
        monitor: Metric to monitor ("val_loss", "val_accuracy").
        mode: "min" for loss, "max" for accuracy.
        save_best_only: Only save when the metric improves.
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value: Optional[float] = None

    def __call__(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, float],
    ) -> Optional[str]:
        """Save checkpoint if conditions are met.

        Args:
            model: Model to save.
            epoch: Current epoch number.
            metrics: Dictionary of current metrics.

        Returns:
            Path to saved checkpoint, or None.
        """
        current = metrics.get(self.monitor)
        if current is None:
            return None

        if self.save_best_only:
            if self.best_value is None:
                improved = True
            elif self.mode == "min":
                improved = current < self.best_value
            else:
                improved = current > self.best_value

            if improved:
                self.best_value = current
                path = self.save_dir / "best.pth"
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                }, str(path))
                return str(path)
            return None
        else:
            path = self.save_dir / f"epoch_{epoch:04d}.pth"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
            }, str(path))
            return str(path)
