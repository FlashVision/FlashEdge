"""PyTorch to OpenVINO IR conversion for Intel hardware."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from flashedge.registry import EXPORTERS


@EXPORTERS.register("openvino")
class OpenVINOExporter:
    """Export PyTorch models to OpenVINO Intermediate Representation (IR).

    Args:
        compress_to_fp16: Whether to compress weights to FP16.
        device: Target inference device (CPU, GPU, VPU).
    """

    def __init__(
        self,
        compress_to_fp16: bool = True,
        device: str = "CPU",
    ) -> None:
        self.compress_to_fp16 = compress_to_fp16
        self.device = device

    def export(
        self,
        model: nn.Module,
        output_dir: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> str:
        """Export a PyTorch model to OpenVINO IR format.

        Conversion path: PyTorch → ONNX → OpenVINO IR (.xml + .bin).

        Args:
            model: The PyTorch model to export.
            output_dir: Directory to save the OpenVINO model files.
            input_shape: Shape of the input tensor.

        Returns:
            Path to the output directory.
        """
        try:
            from openvino.tools import mo
            from openvino.runtime import Core
        except ImportError:
            raise ImportError("OpenVINO export requires openvino: pip install flashedge[openvino]")

        model = model.cpu().eval()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / "model.onnx")
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

            ov_model = mo.convert_model(
                onnx_path,
                compress_to_fp16=self.compress_to_fp16,
            )

            from openvino.runtime import serialize

            xml_path = str(output_dir / "model.xml")
            serialize(ov_model, xml_path)

        print(f"  OpenVINO IR saved: {output_dir}")
        return str(output_dir)

    def export_from_onnx(
        self,
        onnx_path: str,
        output_dir: str,
    ) -> str:
        """Convert an existing ONNX model to OpenVINO IR."""
        try:
            from openvino.tools import mo
            from openvino.runtime import serialize
        except ImportError:
            raise ImportError("OpenVINO export requires openvino: pip install flashedge[openvino]")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ov_model = mo.convert_model(onnx_path, compress_to_fp16=self.compress_to_fp16)
        xml_path = str(output_dir / "model.xml")
        serialize(ov_model, xml_path)

        print(f"  OpenVINO IR saved: {output_dir}")
        return str(output_dir)
