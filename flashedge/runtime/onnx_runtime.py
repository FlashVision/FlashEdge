"""ONNX Runtime inference session wrapper."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

from flashedge.registry import RUNTIMES


@RUNTIMES.register("onnxruntime")
class OnnxInferenceSession:
    """Unified ONNX Runtime inference session for edge deployment.

    Args:
        model_path: Path to the ONNX model file.
        providers: Execution providers (e.g., CPUExecutionProvider, CUDAExecutionProvider).
        session_options: Optional ONNX Runtime session options.
    """

    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None,
        session_options: Optional[Any] = None,
    ) -> None:
        import onnxruntime as ort

        self.model_path = model_path

        if providers is None:
            available = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

        if session_options is None:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 0

        self.session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.input_shapes = [inp.shape for inp in self.session.get_inputs()]
        self.input_dtypes = [inp.type for inp in self.session.get_inputs()]

    def predict(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> List[np.ndarray]:
        """Run inference on the model.

        Args:
            inputs: Input tensor(s) — either a single numpy array (mapped to the
                    first input) or a dict mapping input names to arrays.

        Returns:
            List of output numpy arrays.
        """
        if isinstance(inputs, np.ndarray):
            feed = {self.input_names[0]: inputs.astype(np.float32)}
        else:
            feed = {k: v.astype(np.float32) for k, v in inputs.items()}

        return self.session.run(self.output_names, feed)

    def benchmark(
        self,
        input_shape: Optional[tuple] = None,
        warmup: int = 10,
        runs: int = 100,
    ) -> Dict[str, float]:
        """Benchmark the model's inference latency.

        Args:
            input_shape: Shape of the input tensor. If None, uses the model's input shape.
            warmup: Number of warmup iterations.
            runs: Number of benchmark iterations.

        Returns:
            Dictionary with latency statistics.
        """
        if input_shape is None:
            shape = []
            for dim in self.input_shapes[0]:
                shape.append(dim if isinstance(dim, int) and dim > 0 else 1)
            input_shape = tuple(shape)

        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        feed = {self.input_names[0]: dummy_input}

        for _ in range(warmup):
            self.session.run(self.output_names, feed)

        latencies = []
        for _ in range(runs):
            start = time.perf_counter()
            self.session.run(self.output_names, feed)
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
        """Get model and session metadata."""
        return {
            "model_path": self.model_path,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "input_shapes": self.input_shapes,
            "input_dtypes": self.input_dtypes,
            "providers": self.session.get_providers(),
        }
