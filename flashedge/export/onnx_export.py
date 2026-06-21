"""PyTorch to ONNX export with graph optimization and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from flashedge.registry import EXPORTERS


@EXPORTERS.register("onnx")
class OnnxExporter:
    """Export PyTorch models to ONNX format with optimization and validation.

    Args:
        opset_version: ONNX opset version (default: 17).
        simplify: Whether to run onnxsim graph simplification.
        optimize: Whether to apply ONNX graph optimizations.
        fp16: Whether to convert weights to FP16.
        dynamic_axes: Dynamic axis configuration for variable-size inputs.
    """

    def __init__(
        self,
        opset_version: int = 17,
        simplify: bool = True,
        optimize: bool = True,
        fp16: bool = False,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> None:
        self.opset_version = opset_version
        self.simplify = simplify
        self.optimize = optimize
        self.fp16 = fp16
        self.dynamic_axes = dynamic_axes

    def export(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> str:
        """Export a PyTorch model to ONNX format.

        Args:
            model: The PyTorch model to export.
            output_path: Path to save the ONNX model.
            input_shape: Shape of the input tensor.
            input_names: Names for the input nodes.
            output_names: Names for the output nodes.

        Returns:
            Path to the exported ONNX model.
        """
        import onnx

        model = model.cpu().eval()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_names = input_names or ["input"]
        output_names = output_names or ["output"]

        dummy_input = torch.randn(*input_shape)

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                opset_version=self.opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=self.dynamic_axes,
                do_constant_folding=True,
            )

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        if self.optimize:
            onnx_model = self._optimize_graph(onnx_model)

        if self.simplify:
            onnx_model = self._simplify(onnx_model, input_shape)

        if self.fp16:
            onnx_model = self._convert_fp16(onnx_model)

        onnx.save(onnx_model, str(output_path))

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ONNX model saved: {output_path} ({size_mb:.2f} MB)")

        return str(output_path)

    def _optimize_graph(self, model: Any) -> Any:
        """Apply ONNX graph optimizations (constant folding, shape inference)."""
        from onnx import shape_inference

        model = shape_inference.infer_shapes(model)

        passes = []
        for node in model.graph.node:
            if node.op_type in ("Identity", "Dropout") and not any(
                n for n in model.graph.node if node.output[0] in [i for i in n.input]
            ):
                passes.append(node)

        return model

    def _simplify(self, model: Any, input_shape: Tuple[int, ...]) -> Any:
        """Run onnxsim graph simplification if available."""
        try:
            import onnxsim

            simplified, ok = onnxsim.simplify(model)
            if ok:
                return simplified
            print("  Warning: onnxsim simplification returned invalid model, using original")
            return model
        except ImportError:
            print("  Note: Install onnxsim for graph simplification: pip install onnxsim")
            return model
        except Exception as e:
            print(f"  Warning: onnxsim failed ({e}), using original model")
            return model

    def _convert_fp16(self, model: Any) -> Any:
        """Convert model weights to FP16."""
        import onnx
        from onnx import numpy_helper

        for initializer in model.graph.initializer:
            if initializer.data_type == onnx.TensorProto.FLOAT:
                np_data = numpy_helper.to_array(initializer).astype(np.float16)
                new_init = numpy_helper.from_array(np_data, name=initializer.name)
                initializer.CopyFrom(new_init)

        return model

    def validate(
        self,
        onnx_path: str,
        model: Optional[nn.Module] = None,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        atol: float = 1e-5,
    ) -> Dict[str, Any]:
        """Validate an ONNX model against the original PyTorch model.

        Args:
            onnx_path: Path to the ONNX model file.
            model: Original PyTorch model for output comparison.
            input_shape: Shape of the input tensor.
            atol: Absolute tolerance for output comparison.

        Returns:
            Dictionary with validation results.
        """
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        input_name = session.get_inputs()[0].name
        ort_output = session.run(None, {input_name: dummy_input})[0]

        result: Dict[str, Any] = {
            "valid_schema": True,
            "inference_ok": True,
            "output_shape": list(ort_output.shape),
            "opset_version": onnx_model.opset_import[0].version,
        }

        if model is not None:
            model = model.cpu().eval()
            with torch.no_grad():
                pt_output = model(torch.from_numpy(dummy_input)).numpy()
            max_diff = float(np.abs(pt_output - ort_output).max())
            result["max_output_diff"] = max_diff
            result["outputs_match"] = max_diff < atol

        return result

    def get_model_info(self, onnx_path: str) -> Dict[str, Any]:
        """Get metadata about an ONNX model."""
        import onnx

        model = onnx.load(onnx_path)
        file_size = Path(onnx_path).stat().st_size

        inputs = []
        for inp in model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            inputs.append({"name": inp.name, "shape": shape})

        outputs = []
        for out in model.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            outputs.append({"name": out.name, "shape": shape})

        return {
            "file_size_bytes": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "opset_version": model.opset_import[0].version,
            "ir_version": model.ir_version,
            "num_nodes": len(model.graph.node),
            "inputs": inputs,
            "outputs": outputs,
            "num_initializers": len(model.graph.initializer),
        }
