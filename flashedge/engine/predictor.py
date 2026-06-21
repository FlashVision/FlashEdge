"""Inference predictor supporting multiple model formats."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class Predictor:
    """Unified predictor for PyTorch and ONNX model inference.

    Args:
        model_path: Path to model file (.pth, .pt, .onnx).
        model: Alternatively, a PyTorch nn.Module directly.
        device: Inference device.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.format = "pytorch"

        if model is not None:
            self.model = model.to(device).eval()
        elif model_path is not None:
            self.model_path = Path(model_path)
            self._load_model()
        else:
            raise ValueError("Provide either model_path or model")

    def _load_model(self) -> None:
        suffix = self.model_path.suffix.lower()

        if suffix == ".onnx":
            self.format = "onnx"
            from flashedge.runtime.onnx_runtime import OnnxInferenceSession

            providers = ["CPUExecutionProvider"]
            if self.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.session = OnnxInferenceSession(str(self.model_path), providers=providers)
            self.model = None

        elif suffix in (".pth", ".pt"):
            self.format = "pytorch"
            checkpoint = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
            if isinstance(checkpoint, nn.Module):
                self.model = checkpoint.to(self.device).eval()
            elif isinstance(checkpoint, dict) and "model" in checkpoint:
                self.model = checkpoint["model"].to(self.device).eval()
            else:
                raise ValueError(f"Cannot load model from {self.model_path}")
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Run inference.

        Args:
            x: Input tensor or numpy array.

        Returns:
            Model output.
        """
        if self.format == "onnx":
            if isinstance(x, torch.Tensor):
                x = x.numpy()
            return self.session.predict(x)[0]
        else:
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            with torch.no_grad():
                return self.model(x.to(self.device))

    def benchmark(
        self,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        warmup: int = 10,
        runs: int = 100,
    ) -> Dict[str, float]:
        """Benchmark inference latency.

        Args:
            input_shape: Shape of the input tensor.
            warmup: Number of warmup iterations.
            runs: Number of benchmark iterations.

        Returns:
            Dictionary with latency statistics.
        """
        if self.format == "onnx":
            return self.session.benchmark(input_shape=input_shape, warmup=warmup, runs=runs)

        model = self.model
        dummy = torch.randn(*input_shape, device=self.device)

        with torch.no_grad():
            for _ in range(warmup):
                model(dummy)

            if self.device == "cuda":
                torch.cuda.synchronize()

            latencies = []
            for _ in range(runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                model(dummy)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)

        arr = np.array(latencies)
        return {
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "fps": float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
        }
