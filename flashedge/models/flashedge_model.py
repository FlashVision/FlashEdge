"""FlashEdge — unified model wrapper for edge deployment workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn


class FlashEdge:
    """High-level model wrapper for edge AI workflows.

    Loads a PyTorch model and provides methods for export, profiling,
    optimization, and inference.

    Args:
        model_or_path: A PyTorch nn.Module, or a path to a saved model (.pth/.pt).
        device: Device for inference (cpu, cuda).
    """

    def __init__(
        self,
        model_or_path: Union[str, nn.Module],
        device: str = "cpu",
    ) -> None:
        self.device = device

        if isinstance(model_or_path, nn.Module):
            self.model = model_or_path.to(device).eval()
            self.model_path = None
        elif isinstance(model_or_path, str):
            self.model_path = model_or_path
            self.model = self._load_model(model_or_path, device)
        else:
            raise TypeError(f"Expected nn.Module or str, got {type(model_or_path)}")

    def _load_model(self, path: str, device: str) -> nn.Module:
        """Load a model from a checkpoint file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(str(path), map_location=device, weights_only=False)

        if isinstance(checkpoint, nn.Module):
            return checkpoint.to(device).eval()

        if isinstance(checkpoint, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                if key in checkpoint:
                    model_data = checkpoint[key]
                    if isinstance(model_data, nn.Module):
                        return model_data.to(device).eval()
                    raise ValueError(
                        f"Checkpoint contains '{key}' but it's a state_dict. Provide the model architecture separately."
                    )

        raise ValueError("Cannot load model — provide an nn.Module or a checkpoint containing the full model.")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference."""
        return self.predict(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on input tensor.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(x.to(self.device))

    def export_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        **kwargs: Any,
    ) -> str:
        """Export to ONNX format."""
        from flashedge.export.onnx_export import OnnxExporter

        exporter = OnnxExporter(**kwargs)
        return exporter.export(self.model, output_path, input_shape=input_shape)

    def export_tflite(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        **kwargs: Any,
    ) -> str:
        """Export to TFLite format."""
        from flashedge.export.tflite_export import TFLiteExporter

        exporter = TFLiteExporter(**kwargs)
        return exporter.export(self.model, output_path, input_shape=input_shape)

    def export_torchscript(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        **kwargs: Any,
    ) -> str:
        """Export to TorchScript format."""
        from flashedge.export.torchscript import TorchScriptExporter

        exporter = TorchScriptExporter(**kwargs)
        return exporter.export(self.model, output_path, input_shape=input_shape)

    def profile_latency(
        self,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        runs: int = 100,
    ) -> Dict[str, float]:
        """Profile inference latency."""
        from flashedge.profiling.latency import LatencyProfiler

        profiler = LatencyProfiler(device=self.device)
        return profiler.profile(self.model, input_shape=input_shape, runs=runs)

    def profile_memory(
        self,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> Dict[str, Any]:
        """Profile memory footprint."""
        from flashedge.profiling.memory import MemoryProfiler

        profiler = MemoryProfiler(device=self.device)
        return profiler.profile(self.model, input_shape=input_shape)

    def count_flops(
        self,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> Dict[str, Any]:
        """Count FLOPs and MACs."""
        from flashedge.profiling.flops import FLOPsCounter

        counter = FLOPsCounter()
        return counter.count(self.model, input_shape=input_shape)

    def quantize(self, **kwargs: Any) -> nn.Module:
        """Quantize the model to INT8."""
        from flashedge.optimization.quantization import INT8Quantizer

        quantizer = INT8Quantizer(**kwargs)
        self.model = quantizer.quantize(self.model)
        return self.model

    def prune(self, sparsity: float = 0.3, **kwargs: Any) -> nn.Module:
        """Prune the model."""
        from flashedge.optimization.pruning import StructuredPruner

        pruner = StructuredPruner(sparsity=sparsity, **kwargs)
        self.model = pruner.prune(self.model)
        return self.model

    @property
    def num_parameters(self) -> int:
        """Total number of model parameters."""
        return sum(p.numel() for p in self.model.parameters())

    @property
    def model_size_mb(self) -> float:
        """Model size in megabytes."""
        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_bytes + buffer_bytes) / (1024 * 1024)
