"""Unified model exporter supporting multiple edge formats."""

from __future__ import annotations

from typing import Any, Tuple

import torch.nn as nn


class Exporter:
    """Unified exporter for converting PyTorch models to edge formats.

    Args:
        model: PyTorch model to export.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

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

    def export_coreml(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        **kwargs: Any,
    ) -> str:
        """Export to CoreML format."""
        from flashedge.export.coreml_export import CoreMLExporter

        exporter = CoreMLExporter(**kwargs)
        return exporter.export(self.model, output_path, input_shape=input_shape)

    def export_openvino(
        self,
        output_dir: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        **kwargs: Any,
    ) -> str:
        """Export to OpenVINO IR format."""
        from flashedge.export.openvino_export import OpenVINOExporter

        exporter = OpenVINOExporter(**kwargs)
        return exporter.export(self.model, output_dir, input_shape=input_shape)

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

    def export(
        self,
        output_path: str,
        format: str = "onnx",
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        **kwargs: Any,
    ) -> str:
        """Export model to specified format.

        Args:
            output_path: Path to save the exported model.
            format: Target format (onnx, tflite, coreml, openvino, torchscript).
            input_shape: Shape of the input tensor.

        Returns:
            Path to the exported model.
        """
        dispatch = {
            "onnx": self.export_onnx,
            "tflite": self.export_tflite,
            "coreml": self.export_coreml,
            "openvino": self.export_openvino,
            "torchscript": self.export_torchscript,
        }

        fn = dispatch.get(format)
        if fn is None:
            raise ValueError(f"Unsupported format '{format}'. Available: {list(dispatch.keys())}")

        return fn(output_path, input_shape=input_shape, **kwargs)
