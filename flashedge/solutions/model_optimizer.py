"""Model optimization solution — combines quantization, pruning, and export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


class ModelOptimizer:
    """High-level model optimization pipeline for edge deployment.

    Combines quantization, pruning, and export into a single workflow.
    """

    def optimize(
        self,
        model_or_path: Union[str, nn.Module],
        output_path: str = "optimized_model.onnx",
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        quantize: bool = True,
        prune: bool = False,
        prune_sparsity: float = 0.3,
        export_format: str = "onnx",
    ) -> Dict[str, Any]:
        """Run the full optimization pipeline.

        Args:
            model_or_path: PyTorch model or path to checkpoint.
            output_path: Path to save the optimized model.
            input_shape: Shape of the input tensor.
            quantize: Whether to apply INT8 quantization.
            prune: Whether to apply structured pruning.
            prune_sparsity: Pruning sparsity ratio.
            export_format: Target export format.

        Returns:
            Dictionary with optimization results.
        """
        if isinstance(model_or_path, str):
            checkpoint = torch.load(model_or_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, nn.Module):
                model = checkpoint
            elif isinstance(checkpoint, dict) and "model" in checkpoint:
                model = checkpoint["model"]
            else:
                raise ValueError(f"Cannot load model from {model_or_path}")
        else:
            model = model_or_path

        model = model.cpu().eval()
        results: Dict[str, Any] = {}

        original_params = sum(p.numel() for p in model.parameters())
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        results["original"] = {"params": original_params, "size_mb": original_size}

        if prune:
            from flashedge.optimization.pruning import StructuredPruner

            pruner = StructuredPruner(sparsity=prune_sparsity)
            model = pruner.prune(model)
            results["pruning"] = {"sparsity": prune_sparsity, "applied": True}
        else:
            results["pruning"] = {"applied": False}

        if export_format == "onnx":
            from flashedge.export.onnx_export import OnnxExporter

            exporter = OnnxExporter(opset_version=17, simplify=True)
            export_path = exporter.export(model, output_path, input_shape=input_shape)

            if quantize:
                quantized_path = str(Path(output_path).with_stem(Path(output_path).stem + "_int8"))
                from flashedge.optimization.quantization import INT8Quantizer

                quantizer = INT8Quantizer()
                quantizer.quantize_onnx(export_path, quantized_path)
                results["quantization"] = {"dtype": "int8", "output": quantized_path}
            else:
                results["quantization"] = {"applied": False}

            results["export"] = {"format": "onnx", "output": export_path}
        else:
            from flashedge.engine.exporter import Exporter

            exporter = Exporter(model)
            export_path = exporter.export(output_path, format=export_format, input_shape=input_shape)
            results["export"] = {"format": export_format, "output": export_path}

        return results

    def quantize(
        self,
        model_or_path: str,
        output: str,
        dtype: str = "int8",
        **kwargs: Any,
    ) -> str:
        """Quantize an ONNX model.

        Args:
            model_or_path: Path to the ONNX model.
            output: Path to save the quantized model.
            dtype: Quantization dtype.

        Returns:
            Path to the quantized model.
        """
        from flashedge.optimization.quantization import INT8Quantizer

        quantizer = INT8Quantizer(**kwargs)
        return quantizer.quantize_onnx(model_or_path, output)
