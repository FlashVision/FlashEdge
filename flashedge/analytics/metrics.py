"""Edge-specific metrics: latency, throughput, model size, accuracy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class EdgeMetrics:
    """Collection of edge-relevant metrics for model evaluation.

    Provides methods to measure latency, throughput, model size,
    accuracy, and compute efficiency scores.
    """

    @staticmethod
    def latency_stats(latencies: List[float]) -> Dict[str, float]:
        """Compute latency statistics from a list of measurements.

        Args:
            latencies: List of latency measurements in milliseconds.

        Returns:
            Dictionary with statistical summary.
        """
        arr = np.array(latencies)
        return {
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "fps": float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
        }

    @staticmethod
    def model_size(model: nn.Module) -> Dict[str, float]:
        """Compute model size metrics.

        Args:
            model: PyTorch model.

        Returns:
            Dictionary with size metrics.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "param_size_mb": param_bytes / (1024 * 1024),
            "buffer_size_mb": buffer_bytes / (1024 * 1024),
            "total_size_mb": (param_bytes + buffer_bytes) / (1024 * 1024),
        }

    @staticmethod
    def file_size(path: str) -> Dict[str, float]:
        """Get file size metrics for an exported model.

        Args:
            path: Path to model file.

        Returns:
            Dictionary with file size information.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        size = p.stat().st_size
        return {
            "file_size_bytes": size,
            "file_size_kb": size / 1024,
            "file_size_mb": size / (1024 * 1024),
        }

    @staticmethod
    @torch.no_grad()
    def accuracy(
        model: nn.Module,
        data_loader: Any,
        device: str = "cpu",
        topk: Tuple[int, ...] = (1, 5),
    ) -> Dict[str, float]:
        """Compute top-k accuracy on a dataset.

        Args:
            model: PyTorch model.
            data_loader: Data loader yielding (images, targets).
            device: Inference device.
            topk: Tuple of k values for top-k accuracy.

        Returns:
            Dictionary with top-k accuracy values.
        """
        model = model.to(device).eval()
        correct = {k: 0 for k in topk}
        total = 0

        for batch in data_loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                images, targets = batch[0].to(device), batch[1].to(device)
            else:
                continue

            outputs = model(images)
            max_k = min(max(topk), outputs.size(1))
            _, topk_pred = outputs.topk(max_k, dim=1)

            for k in topk:
                effective_k = min(k, max_k)
                for i in range(targets.size(0)):
                    if targets[i].item() in topk_pred[i, :effective_k].tolist():
                        correct[k] += 1

            total += targets.size(0)

        result = {}
        for k in topk:
            result[f"top{k}_accuracy"] = correct[k] / max(total, 1)
        result["num_samples"] = total

        return result

    @staticmethod
    def compute_sparsity(model: nn.Module) -> Dict[str, float]:
        """Compute model weight sparsity.

        Args:
            model: PyTorch model.

        Returns:
            Dictionary with sparsity metrics.
        """
        total_params = 0
        zero_params = 0

        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

        sparsity = zero_params / max(total_params, 1)
        return {
            "total_params": total_params,
            "zero_params": zero_params,
            "sparsity": sparsity,
            "density": 1.0 - sparsity,
        }

    @staticmethod
    def efficiency_score(
        accuracy: float,
        latency_ms: float,
        model_size_mb: float,
        target_accuracy: float = 0.7,
        target_latency_ms: float = 10.0,
        target_size_mb: float = 5.0,
    ) -> float:
        """Compute a composite efficiency score for edge deployment.

        Higher scores indicate better suitability for edge deployment.
        Score range: 0.0 to 1.0.

        Args:
            accuracy: Model accuracy (0–1).
            latency_ms: Mean inference latency.
            model_size_mb: Model file size.
            target_accuracy: Target accuracy threshold.
            target_latency_ms: Target latency threshold.
            target_size_mb: Target model size threshold.

        Returns:
            Composite efficiency score.
        """
        acc_score = min(accuracy / target_accuracy, 1.0)
        lat_score = min(target_latency_ms / max(latency_ms, 0.01), 1.0)
        size_score = min(target_size_mb / max(model_size_mb, 0.01), 1.0)

        score = 0.4 * acc_score + 0.35 * lat_score + 0.25 * size_score
        return min(max(score, 0.0), 1.0)
