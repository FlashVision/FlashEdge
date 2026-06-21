"""Edge deployment solution — one-click model export and packaging."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

import torch.nn as nn


class EdgeDeployer:
    """Deploy models to edge devices with runtime packaging.

    Handles export, optimization, validation, and deployment artifact
    creation for target edge runtimes.
    """

    def deploy(
        self,
        model_or_path: str,
        target: str = "onnxruntime",
        output_dir: str = "deployed",
        input_shape: tuple = (1, 3, 224, 224),
        quantize: bool = False,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Deploy a model to the target edge runtime.

        Args:
            model_or_path: Path to model file or PyTorch model.
            target: Target runtime (onnxruntime, tflite, openvino, coreml).
            output_dir: Output directory for deployment artifacts.
            input_shape: Input tensor shape.
            quantize: Whether to apply INT8 quantization.
            validate: Whether to validate the exported model.

        Returns:
            Dictionary with deployment results.
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        source_path = Path(model_or_path)

        result: Dict[str, Any] = {
            "target": target,
            "output_dir": str(output),
            "input_shape": list(input_shape),
            "quantized": quantize,
        }

        if target == "onnxruntime":
            deployed_path = self._deploy_onnx(source_path, output, input_shape, quantize, validate)
        elif target == "tflite":
            deployed_path = self._deploy_tflite(source_path, output, input_shape, quantize)
        elif target == "openvino":
            deployed_path = self._deploy_openvino(source_path, output, input_shape)
        elif target == "coreml":
            deployed_path = self._deploy_coreml(source_path, output, input_shape)
        else:
            raise ValueError(f"Unsupported target: {target}. Use: onnxruntime, tflite, openvino, coreml")

        result["model_path"] = deployed_path

        manifest = {
            "target_runtime": target,
            "model_file": Path(deployed_path).name,
            "input_shape": list(input_shape),
            "quantized": quantize,
        }
        with open(output / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        result["manifest"] = str(output / "manifest.json")
        print(f"  Deployment complete: {output}")

        return result

    def _deploy_onnx(
        self,
        source: Path,
        output: Path,
        input_shape: tuple,
        quantize: bool,
        validate: bool,
    ) -> str:
        """Deploy as ONNX Runtime package."""
        if source.suffix == ".onnx":
            dest = output / source.name
            shutil.copy2(str(source), str(dest))
        else:
            dest = output / "model.onnx"
            model = self._load_pytorch_model(source)
            from flashedge.export.onnx_export import OnnxExporter

            exporter = OnnxExporter(opset_version=17, simplify=True)
            exporter.export(model, str(dest), input_shape=input_shape)

        if quantize:
            quantized_path = output / "model_int8.onnx"
            from flashedge.optimization.quantization import INT8Quantizer

            quantizer = INT8Quantizer()
            quantizer.quantize_onnx(str(dest), str(quantized_path))
            return str(quantized_path)

        if validate:
            from flashedge.export.onnx_export import OnnxExporter

            exporter = OnnxExporter()
            info = exporter.get_model_info(str(dest))
            print(f"  ONNX model info: {info['num_nodes']} nodes, {info['file_size_mb']:.2f} MB")

        return str(dest)

    def _deploy_tflite(
        self,
        source: Path,
        output: Path,
        input_shape: tuple,
        quantize: bool,
    ) -> str:
        """Deploy as TFLite package."""
        dest = output / "model.tflite"

        if source.suffix == ".tflite":
            shutil.copy2(str(source), str(dest))
        else:
            model = self._load_pytorch_model(source)
            from flashedge.export.tflite_export import TFLiteExporter

            exporter = TFLiteExporter(quantize=quantize, dtype="int8" if quantize else "float32")
            exporter.export(model, str(dest), input_shape=input_shape)

        return str(dest)

    def _deploy_openvino(
        self,
        source: Path,
        output: Path,
        input_shape: tuple,
    ) -> str:
        """Deploy as OpenVINO package."""
        ov_dir = output / "openvino"

        if source.suffix == ".onnx":
            from flashedge.export.openvino_export import OpenVINOExporter

            exporter = OpenVINOExporter()
            exporter.export_from_onnx(str(source), str(ov_dir))
        else:
            model = self._load_pytorch_model(source)
            from flashedge.export.openvino_export import OpenVINOExporter

            exporter = OpenVINOExporter()
            exporter.export(model, str(ov_dir), input_shape=input_shape)

        return str(ov_dir / "model.xml")

    def _deploy_coreml(
        self,
        source: Path,
        output: Path,
        input_shape: tuple,
    ) -> str:
        """Deploy as CoreML package."""
        dest = output / "model.mlpackage"

        model = self._load_pytorch_model(source)
        from flashedge.export.coreml_export import CoreMLExporter

        exporter = CoreMLExporter()
        exporter.export(model, str(dest), input_shape=input_shape)

        return str(dest)

    def _load_pytorch_model(self, path: Path) -> nn.Module:
        """Load a PyTorch model from a checkpoint file."""
        import torch

        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(checkpoint, nn.Module):
            return checkpoint.eval()
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            return checkpoint["model"].eval()
        raise ValueError(f"Cannot load PyTorch model from {path}")
