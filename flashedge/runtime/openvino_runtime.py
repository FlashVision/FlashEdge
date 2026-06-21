"""OpenVINO inference runtime for Intel hardware."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from flashedge.registry import RUNTIMES


@RUNTIMES.register("openvino")
class OpenVINOInferenceSession:
    """OpenVINO inference session for Intel CPU/GPU/VPU deployment.

    Args:
        model_path: Path to the OpenVINO IR model (.xml file).
        device: Target device (CPU, GPU, VPU, MYRIAD).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
    ) -> None:
        try:
            from openvino.runtime import Core
        except ImportError:
            raise ImportError("OpenVINO runtime requires openvino: pip install flashedge[openvino]")

        self.model_path = model_path
        self.device = device

        core = Core()
        model = core.read_model(model_path)
        self.compiled = core.compile_model(model, device)

        self.input_names = [inp.any_name for inp in model.inputs]
        self.output_names = [out.any_name for out in model.outputs]
        self.input_shapes = [list(inp.shape) for inp in model.inputs]

    def predict(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> List[np.ndarray]:
        """Run inference using OpenVINO.

        Args:
            inputs: Input tensor(s).

        Returns:
            List of output numpy arrays.
        """
        if isinstance(inputs, np.ndarray):
            feed = {self.input_names[0]: inputs.astype(np.float32)}
        else:
            feed = {k: v.astype(np.float32) for k, v in inputs.items()}

        result = self.compiled(feed)

        outputs = []
        for i in range(len(self.output_names)):
            outputs.append(result[i].copy())

        return outputs

    def benchmark(
        self,
        input_shape: Optional[tuple] = None,
        warmup: int = 10,
        runs: int = 100,
    ) -> Dict[str, float]:
        """Benchmark OpenVINO model latency.

        Args:
            input_shape: Input tensor shape.
            warmup: Number of warmup iterations.
            runs: Number of benchmark iterations.

        Returns:
            Dictionary with latency statistics.
        """
        if input_shape is None:
            input_shape = tuple(self.input_shapes[0])

        dummy = np.random.randn(*input_shape).astype(np.float32)
        feed = {self.input_names[0]: dummy}

        for _ in range(warmup):
            self.compiled(feed)

        latencies = []
        for _ in range(runs):
            start = time.perf_counter()
            self.compiled(feed)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        latencies_np = np.array(latencies)
        return {
            "mean_ms": float(latencies_np.mean()),
            "std_ms": float(latencies_np.std()),
            "min_ms": float(latencies_np.min()),
            "max_ms": float(latencies_np.max()),
            "p50_ms": float(np.percentile(latencies_np, 50)),
            "p95_ms": float(np.percentile(latencies_np, 95)),
            "p99_ms": float(np.percentile(latencies_np, 99)),
            "fps": float(1000.0 / latencies_np.mean()),
        }

    def get_info(self) -> Dict[str, Any]:
        """Get model and runtime metadata."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "input_shapes": self.input_shapes,
        }
