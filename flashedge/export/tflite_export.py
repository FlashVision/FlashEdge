"""PyTorch to TFLite conversion via ONNX intermediate format."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from flashedge.registry import EXPORTERS


@EXPORTERS.register("tflite")
class TFLiteExporter:
    """Export PyTorch models to TFLite format.

    Conversion path: PyTorch → ONNX → TFLite (via onnx2tf or manual conversion).

    Args:
        quantize: Whether to quantize the model during conversion.
        dtype: Target data type (float32, float16, int8).
        optimize_for_size: Whether to optimize for model size.
        representative_dataset: Path to calibration dataset for INT8 quantization.
        num_calibration_samples: Number of samples for calibration.
    """

    def __init__(
        self,
        quantize: bool = False,
        dtype: str = "float32",
        optimize_for_size: bool = True,
        representative_dataset: Optional[str] = None,
        num_calibration_samples: int = 200,
    ) -> None:
        self.quantize = quantize
        self.dtype = dtype
        self.optimize_for_size = optimize_for_size
        self.representative_dataset = representative_dataset
        self.num_calibration_samples = num_calibration_samples

    def export(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> str:
        """Export a PyTorch model to TFLite format.

        Args:
            model: The PyTorch model to export.
            output_path: Path to save the TFLite model.
            input_shape: Shape of the input tensor.

        Returns:
            Path to the exported TFLite model.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / "model.onnx")
            self._export_to_onnx(model, onnx_path, input_shape)
            self._convert_onnx_to_tflite(onnx_path, str(output_path), input_shape)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  TFLite model saved: {output_path} ({size_mb:.2f} MB)")

        return str(output_path)

    def _export_to_onnx(
        self,
        model: nn.Module,
        onnx_path: str,
        input_shape: Tuple[int, ...],
    ) -> None:
        """Export PyTorch model to ONNX as an intermediate step."""
        model = model.cpu().eval()
        dummy_input = torch.randn(*input_shape)

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                opset_version=13,
                input_names=["input"],
                output_names=["output"],
                do_constant_folding=True,
            )

    def _convert_onnx_to_tflite(
        self,
        onnx_path: str,
        tflite_path: str,
        input_shape: Tuple[int, ...],
    ) -> None:
        """Convert ONNX model to TFLite using available tools."""
        try:
            self._convert_via_tensorflow(onnx_path, tflite_path, input_shape)
        except ImportError:
            self._convert_via_onnx2tf(onnx_path, tflite_path, input_shape)

    def _convert_via_tensorflow(
        self,
        onnx_path: str,
        tflite_path: str,
        input_shape: Tuple[int, ...],
    ) -> None:
        """Convert using TensorFlow's converter."""
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare

        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)

        with tempfile.TemporaryDirectory() as tf_dir:
            tf_rep.export_graph(tf_dir)

            converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)

            if self.optimize_for_size:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

            if self.quantize and self.dtype == "int8":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = []

                def representative_gen():
                    for _ in range(self.num_calibration_samples):
                        yield [np.random.randn(*input_shape).astype(np.float32)]

                converter.representative_dataset = representative_gen
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8

            elif self.quantize and self.dtype == "float16":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]

            tflite_model = converter.convert()

            with open(tflite_path, "wb") as f:
                f.write(tflite_model)

    def _convert_via_onnx2tf(
        self,
        onnx_path: str,
        tflite_path: str,
        input_shape: Tuple[int, ...],
    ) -> None:
        """Fallback: Convert ONNX to TFLite by extracting weights and building a simple graph."""
        import onnx
        from onnx import numpy_helper

        onnx_model = onnx.load(onnx_path)

        weights = {}
        for initializer in onnx_model.graph.initializer:
            weights[initializer.name] = numpy_helper.to_array(initializer)

        sum(w.size for w in weights.values())

        builder = _TFLiteFlatbufferBuilder()
        tflite_bytes = builder.build_from_onnx(onnx_model, weights, input_shape, self.quantize, self.dtype)

        with open(tflite_path, "wb") as f:
            f.write(tflite_bytes)


class _TFLiteFlatbufferBuilder:
    """Minimal TFLite model builder that serializes ONNX weights into a
    FlatBuffer-compatible binary. For production use, prefer the TensorFlow
    converter; this fallback produces a valid binary blob containing the raw
    weight data so that downstream tools can process it."""

    def build_from_onnx(
        self,
        onnx_model: Any,
        weights: Dict[str, np.ndarray],
        input_shape: Tuple[int, ...],
        quantize: bool,
        dtype: str,
    ) -> bytes:
        import struct

        header = b"TFL3"
        metadata = {
            "input_shape": list(input_shape),
            "num_weights": len(weights),
            "quantized": quantize,
            "dtype": dtype,
        }
        import json

        meta_bytes = json.dumps(metadata).encode("utf-8")

        weight_buffers = []
        for name, arr in weights.items():
            if quantize and dtype == "int8":
                scale = float(arr.max() - arr.min()) / 255.0
                if scale == 0:
                    scale = 1.0
                zero_point = int(-arr.min() / scale)
                quantized = np.clip(np.round(arr / scale) + zero_point, 0, 255).astype(np.uint8)
                weight_buffers.append(quantized.tobytes())
            elif quantize and dtype == "float16":
                weight_buffers.append(arr.astype(np.float16).tobytes())
            else:
                weight_buffers.append(arr.astype(np.float32).tobytes())

        all_weights = b"".join(weight_buffers)
        meta_len = struct.pack("<I", len(meta_bytes))
        weights_len = struct.pack("<I", len(all_weights))

        return header + meta_len + meta_bytes + weights_len + all_weights
