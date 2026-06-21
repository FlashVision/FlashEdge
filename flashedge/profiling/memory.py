"""Memory footprint analysis for PyTorch and exported models."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class MemoryProfiler:
    """Profile model memory footprint including parameters, buffers, and peak activation memory.

    Args:
        device: Device for profiling ("cpu" or "cuda").
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def profile(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> Dict[str, Any]:
        """Profile model memory usage.

        Args:
            model: PyTorch model to profile.
            input_shape: Shape of the input tensor.

        Returns:
            Dictionary with memory statistics.
        """
        model = model.to(self.device).eval()

        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_buffers = sum(b.numel() for b in model.buffers())

        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_bytes = param_bytes + buffer_bytes

        activation_bytes = self._estimate_activation_memory(model, input_shape)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            state_dict_size = os.path.getsize(f.name)
            os.unlink(f.name)

        peak_memory = model_size_bytes + activation_bytes

        return {
            "num_parameters": num_params,
            "num_trainable": num_trainable,
            "num_buffers": num_buffers,
            "param_size_mb": param_bytes / (1024 * 1024),
            "buffer_size_mb": buffer_bytes / (1024 * 1024),
            "model_size_mb": model_size_bytes / (1024 * 1024),
            "activation_size_mb": activation_bytes / (1024 * 1024),
            "peak_memory_mb": peak_memory / (1024 * 1024),
            "state_dict_size_mb": state_dict_size / (1024 * 1024),
        }

    def _estimate_activation_memory(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
    ) -> int:
        """Estimate peak activation memory by tracing forward pass."""
        activation_sizes: list = []
        hooks = []

        def hook_fn(module: nn.Module, inp: Any, out: Any) -> None:
            if isinstance(out, torch.Tensor):
                activation_sizes.append(out.numel() * out.element_size())
            elif isinstance(out, (tuple, list)):
                for o in out:
                    if isinstance(o, torch.Tensor):
                        activation_sizes.append(o.numel() * o.element_size())

        for module in model.modules():
            hooks.append(module.register_forward_hook(hook_fn))

        dummy = torch.randn(*input_shape, device=self.device)
        with torch.no_grad():
            model(dummy)

        for h in hooks:
            h.remove()

        return max(activation_sizes) if activation_sizes else 0

    def profile_file(self, model_path: str) -> Dict[str, float]:
        """Profile an exported model file's disk size.

        Args:
            model_path: Path to model file (ONNX, TFLite, etc.).

        Returns:
            Dictionary with file size information.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        size_bytes = path.stat().st_size
        return {
            "file_path": str(path),
            "file_size_bytes": size_bytes,
            "file_size_kb": size_bytes / 1024,
            "file_size_mb": size_bytes / (1024 * 1024),
        }

    def compare_models(
        self,
        models: Dict[str, nn.Module],
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> Dict[str, Dict[str, Any]]:
        """Compare memory footprint across multiple models.

        Args:
            models: Dictionary mapping model names to PyTorch models.
            input_shape: Shape of the input tensor.

        Returns:
            Dictionary mapping model names to their memory profiles.
        """
        results = {}
        for name, model in models.items():
            results[name] = self.profile(model, input_shape)
        return results
