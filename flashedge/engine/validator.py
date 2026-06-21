"""Model validation and accuracy evaluation for edge models."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class Validator:
    """Validate edge models for accuracy and performance.

    Args:
        device: Validation device.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        val_loader: Any,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """Run full validation on a dataset.

        Args:
            model: Model to validate.
            val_loader: Validation data loader.
            criterion: Loss function.

        Returns:
            Dictionary with validation metrics.
        """
        model = model.to(self.device).eval()
        criterion = criterion or nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0
        top5_correct = 0

        for batch in val_loader:
            if isinstance(batch, (tuple, list)):
                images, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                continue

            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            _, top5_pred = outputs.topk(min(5, outputs.size(1)), dim=1)
            top5_correct += sum(targets[i].item() in top5_pred[i].tolist() for i in range(targets.size(0)))

            total += targets.size(0)

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
            "top5_accuracy": top5_correct / max(total, 1),
            "num_samples": total,
        }

    def compare_models(
        self,
        models: Dict[str, nn.Module],
        val_loader: Any,
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models on the same dataset.

        Args:
            models: Dictionary mapping model names to PyTorch models.
            val_loader: Validation data loader.

        Returns:
            Dictionary mapping model names to their validation metrics.
        """
        results = {}
        for name, model in models.items():
            print(f"  Validating: {name}")
            results[name] = self.validate(model, val_loader)
        return results

    @staticmethod
    def model_size_mb(model: nn.Module) -> float:
        """Get model size in megabytes."""
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_bytes + buffer_bytes) / (1024 * 1024)

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
