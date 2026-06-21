"""TFLite interpreter wrapper for on-device inference."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

from flashedge.registry import RUNTIMES


@RUNTIMES.register("tflite")
class TFLiteInferenceSession:
    """TFLite inference session for mobile and embedded deployment.

    Supports both the full TensorFlow Lite and the tflite-runtime package.

    Args:
        model_path: Path to the TFLite model file.
        num_threads: Number of CPU threads for inference.
    """

    def __init__(
        self,
        model_path: str,
        num_threads: int = 4,
    ) -> None:
        self.model_path = model_path
        self.interpreter = self._load_interpreter(model_path, num_threads)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _load_interpreter(self, model_path: str, num_threads: int) -> Any:
        """Load TFLite interpreter from available packages."""
        try:
            from tflite_runtime.interpreter import Interpreter

            return Interpreter(model_path=model_path, num_threads=num_threads)
        except ImportError:
            pass

        try:
            import tensorflow as tf

            return tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        except ImportError:
            raise ImportError(
                "TFLite inference requires either tflite-runtime or tensorflow: pip install flashedge[tflite]"
            )

    def predict(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> List[np.ndarray]:
        """Run inference on the TFLite model.

        Args:
            inputs: Input tensor(s).

        Returns:
            List of output numpy arrays.
        """
        if isinstance(inputs, np.ndarray):
            input_data = inputs
        else:
            input_name = self.input_details[0]["name"]
            input_data = inputs.get(input_name, list(inputs.values())[0])

        expected_dtype = self.input_details[0]["dtype"]
        input_data = input_data.astype(expected_dtype)

        expected_shape = self.input_details[0]["shape"]
        if list(input_data.shape) != list(expected_shape):
            self.interpreter.resize_tensor_input(self.input_details[0]["index"], list(input_data.shape))
            self.interpreter.allocate_tensors()

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        outputs = []
        for detail in self.output_details:
            outputs.append(self.interpreter.get_tensor(detail["index"]).copy())

        return outputs

    def benchmark(
        self,
        input_shape: Optional[tuple] = None,
        warmup: int = 10,
        runs: int = 100,
    ) -> Dict[str, float]:
        """Benchmark TFLite model latency.

        Args:
            input_shape: Input tensor shape. If None, uses the model's input shape.
            warmup: Number of warmup iterations.
            runs: Number of benchmark iterations.

        Returns:
            Dictionary with latency statistics.
        """
        if input_shape is None:
            input_shape = tuple(self.input_details[0]["shape"])

        dtype = self.input_details[0]["dtype"]
        dummy = np.random.randn(*input_shape).astype(dtype)

        for _ in range(warmup):
            self.predict(dummy)

        latencies = []
        for _ in range(runs):
            start = time.perf_counter()
            self.predict(dummy)
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
        """Get model metadata."""
        return {
            "model_path": self.model_path,
            "num_inputs": len(self.input_details),
            "num_outputs": len(self.output_details),
            "input_shapes": [list(d["shape"]) for d in self.input_details],
            "input_dtypes": [str(d["dtype"]) for d in self.input_details],
            "output_shapes": [list(d["shape"]) for d in self.output_details],
        }
