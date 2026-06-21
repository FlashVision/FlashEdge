"""Latency profiling for CPU, GPU, and NPU targets.

Supports profiling both PyTorch and ONNX Runtime models with detailed
percentile statistics and per-layer breakdown.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class LatencyProfiler:
    """Profile model inference latency on target hardware.

    Args:
        device: Target device — "cpu" or "cuda".
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def profile(
        self,
        model_or_path: Union[str, nn.Module],
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        warmup: int = 10,
        runs: int = 100,
    ) -> Dict[str, float]:
        """Profile inference latency.

        Args:
            model_or_path: PyTorch model or path to ONNX model file.
            input_shape: Shape of the input tensor.
            warmup: Number of warmup iterations.
            runs: Number of benchmark iterations.

        Returns:
            Dictionary with latency statistics (all values in milliseconds).
        """
        if isinstance(model_or_path, str):
            path = Path(model_or_path)
            if path.suffix == ".onnx":
                return self._profile_onnx(str(path), input_shape, warmup, runs)
            elif path.suffix == ".tflite":
                return self._profile_tflite(str(path), input_shape, warmup, runs)
            else:
                model = torch.load(str(path), map_location=self.device, weights_only=False)
                if isinstance(model, dict) and "model" in model:
                    model = model["model"]
                return self._profile_pytorch(model, input_shape, warmup, runs)
        else:
            return self._profile_pytorch(model_or_path, input_shape, warmup, runs)

    def _profile_pytorch(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        warmup: int,
        runs: int,
    ) -> Dict[str, float]:
        """Profile a PyTorch model."""
        model = model.to(self.device).eval()
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

        return self._compute_stats(latencies)

    def _profile_onnx(
        self,
        onnx_path: str,
        input_shape: Tuple[int, ...],
        warmup: int,
        runs: int,
    ) -> Dict[str, float]:
        """Profile an ONNX model using ONNX Runtime."""
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(*input_shape).astype(np.float32)

        for _ in range(warmup):
            session.run(None, {input_name: dummy})

        latencies = []
        for _ in range(runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy})
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        return self._compute_stats(latencies)

    def _profile_tflite(
        self,
        tflite_path: str,
        input_shape: Tuple[int, ...],
        warmup: int,
        runs: int,
    ) -> Dict[str, float]:
        """Profile a TFLite model."""
        from flashedge.runtime.tflite_runtime import TFLiteInferenceSession

        session = TFLiteInferenceSession(tflite_path)
        return session.benchmark(input_shape=input_shape, warmup=warmup, runs=runs)

    def profile_per_layer(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        runs: int = 10,
    ) -> List[Dict[str, Any]]:
        """Profile per-layer latency breakdown for a PyTorch model.

        Args:
            model: PyTorch model.
            input_shape: Input tensor shape.
            runs: Number of profiling runs.

        Returns:
            List of dicts with per-layer timing information.
        """
        model = model.to(self.device).eval()
        layer_times: Dict[str, List[float]] = {}
        hooks = []

        def make_hook(name: str):
            def hook_fn(module: nn.Module, inp: Any, out: Any) -> None:
                if self.device == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - hook_fn._start) * 1000
                layer_times.setdefault(name, []).append(elapsed)

            def pre_hook_fn(module: nn.Module, inp: Any) -> None:
                if self.device == "cuda":
                    torch.cuda.synchronize()
                hook_fn._start = time.perf_counter()

            return pre_hook_fn, hook_fn

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.AdaptiveAvgPool2d)):
                pre_fn, post_fn = make_hook(name)
                hooks.append(module.register_forward_pre_hook(pre_fn))
                hooks.append(module.register_forward_hook(post_fn))

        dummy = torch.randn(*input_shape, device=self.device)
        with torch.no_grad():
            for _ in range(runs):
                model(dummy)

        for h in hooks:
            h.remove()

        results = []
        for name, times in layer_times.items():
            times_np = np.array(times)
            results.append({
                "layer": name,
                "mean_ms": float(times_np.mean()),
                "std_ms": float(times_np.std()),
                "percentage": 0.0,
            })

        total = sum(r["mean_ms"] for r in results)
        for r in results:
            r["percentage"] = (r["mean_ms"] / total * 100) if total > 0 else 0.0

        results.sort(key=lambda x: x["mean_ms"], reverse=True)
        return results

    def _compute_stats(self, latencies: List[float]) -> Dict[str, float]:
        """Compute statistical summary of latency measurements."""
        arr = np.array(latencies)
        return {
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "fps": float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
        }
