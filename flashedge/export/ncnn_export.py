"""PyTorch to NCNN export via ONNX intermediate representation.

Converts PyTorch models to NCNN param/bin format, with an optional
Python-based fallback converter when the ``onnx2ncnn`` CLI tool is not
installed.
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from flashedge.registry import EXPORTERS

_ONNX_TO_NCNN_OP: Dict[str, str] = {
    "Conv": "Convolution",
    "Relu": "ReLU",
    "MaxPool": "Pooling",
    "AveragePool": "Pooling",
    "GlobalAveragePool": "Pooling",
    "Flatten": "Flatten",
    "Gemm": "InnerProduct",
    "Add": "BinaryOp",
    "BatchNormalization": "BatchNorm",
    "Reshape": "Reshape",
    "Softmax": "Softmax",
    "Sigmoid": "Sigmoid",
    "Clip": "ReLU",
    "Concat": "Concat",
    "Transpose": "Permute",
    "MatMul": "InnerProduct",
    "Unsqueeze": "ExpandDims",
}


@EXPORTERS.register("ncnn")
class NCNNExporter:
    """Export PyTorch models to NCNN param/bin format via ONNX.

    The pipeline is PyTorch → ONNX → (optional simplification) → NCNN.
    If the ``onnx2ncnn`` CLI binary is on ``$PATH`` it is used directly;
    otherwise a pure-Python converter handles the common operator subset.

    Args:
        optimize: Simplify the intermediate ONNX graph with ``onnxsim``.
        fp16: Store NCNN weights in FP16.
        input_names: ONNX input node names.
        output_names: ONNX output node names.
        opset_version: ONNX opset version for the intermediate export.
    """

    def __init__(
        self,
        optimize: bool = True,
        fp16: bool = False,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        opset_version: int = 13,
    ) -> None:
        self.optimize = optimize
        self.fp16 = fp16
        self.input_names = input_names or ["input"]
        self.output_names = output_names or ["output"]
        self.opset_version = opset_version

    def export(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> str:
        """Export a PyTorch model to NCNN param/bin files.

        Args:
            model: The PyTorch model to export.
            output_path: Destination path prefix (e.g. ``"out/model"``).
                         ``.param`` and ``.bin`` extensions are appended automatically.
            input_shape: Shape of the dummy input tensor.

        Returns:
            Path to the generated ``.param`` file.
        """
        import onnx

        model = model.cpu().eval()
        output_prefix = str(Path(output_path).with_suffix(""))
        Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            onnx_path = tmp.name

        try:
            dummy_input = torch.randn(*input_shape)
            print(f"  Exporting to intermediate ONNX (opset {self.opset_version})...")
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    opset_version=self.opset_version,
                    input_names=self.input_names,
                    output_names=self.output_names,
                    do_constant_folding=True,
                )

            if self.optimize:
                onnx_path = self._simplify_onnx(onnx_path)

            param_path, bin_path = self._convert_to_ncnn(onnx_path, output_prefix)
        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

        param_size = Path(param_path).stat().st_size / 1024
        bin_size = Path(bin_path).stat().st_size / (1024 * 1024)
        print(f"  NCNN model saved: {param_path} ({param_size:.1f} KB param, {bin_size:.2f} MB bin)")

        return param_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simplify_onnx(self, onnx_path: str) -> str:
        """Simplify the ONNX graph with ``onnxsim`` if available."""
        try:
            import onnx
            import onnxsim

            onnx_model = onnx.load(onnx_path)
            simplified, ok = onnxsim.simplify(onnx_model)
            if ok:
                onnx.save(simplified, onnx_path)
                print("  ONNX graph simplified with onnxsim")
            else:
                print("  Warning: onnxsim returned invalid model, skipping")
        except ImportError:
            print("  Note: Install onnxsim for graph simplification")
        except Exception as exc:
            print(f"  Warning: onnxsim failed ({exc}), skipping")
        return onnx_path

    def _convert_to_ncnn(self, onnx_path: str, output_prefix: str) -> Tuple[str, str]:
        """Convert ONNX to NCNN, preferring the native CLI tool."""
        try:
            subprocess.run(["onnx2ncnn", "--help"], capture_output=True, check=False)
            param_path = output_prefix + ".param"
            bin_path = output_prefix + ".bin"
            subprocess.run(
                ["onnx2ncnn", onnx_path, param_path, bin_path],
                check=True,
                capture_output=True,
            )
            print("  Converted ONNX → NCNN using onnx2ncnn tool")
            return param_path, bin_path
        except FileNotFoundError:
            print("  onnx2ncnn not found, using Python-based converter")
            return self._onnx_to_ncnn_python(onnx_path, output_prefix)

    def _onnx_to_ncnn_python(self, onnx_path: str, output_prefix: str) -> Tuple[str, str]:
        """Pure-Python ONNX → NCNN converter for the common operator subset.

        Parses the ONNX protobuf graph and writes NCNN-compatible ``.param``
        (text) and ``.bin`` (binary weights) files.

        Args:
            onnx_path: Path to the source ONNX model.
            output_prefix: File path prefix for the output files.

        Returns:
            Tuple of ``(param_path, bin_path)``.
        """
        import onnx
        from onnx import numpy_helper

        onnx_model = onnx.load(onnx_path)
        graph = onnx_model.graph

        initializers: Dict[str, np.ndarray] = {}
        for init in graph.initializer:
            initializers[init.name] = numpy_helper.to_array(init)

        blob_names: set[str] = set()
        for node in graph.node:
            for name in list(node.input) + list(node.output):
                if name and name not in initializers:
                    blob_names.add(name)
        for inp in graph.input:
            blob_names.add(inp.name)
        for out in graph.output:
            blob_names.add(out.name)

        layer_count = len(graph.node)
        blob_count = len(blob_names)

        param_path = output_prefix + ".param"
        bin_path = output_prefix + ".bin"

        bin_file = open(bin_path, "wb")
        try:
            lines: List[str] = ["7767517", f"{layer_count} {blob_count}"]

            for node in graph.node:
                ncnn_op = _ONNX_TO_NCNN_OP.get(node.op_type, node.op_type)
                layer_name = node.name or node.output[0]
                real_inputs = [i for i in node.input if i and i not in initializers]
                real_outputs = [o for o in node.output if o]
                in_count = len(real_inputs)
                out_count = len(real_outputs)

                parts = [ncnn_op, layer_name, str(in_count), str(out_count)]
                parts.extend(real_inputs)
                parts.extend(real_outputs)

                attrs = {a.name: a for a in node.attribute}
                self._write_layer_params(node.op_type, attrs, initializers, node, parts, bin_file)

                lines.append(" ".join(parts))

            with open(param_path, "w") as f:
                f.write("\n".join(lines) + "\n")
        finally:
            bin_file.close()

        return param_path, bin_path

    def _write_layer_params(
        self,
        op_type: str,
        attrs: Dict[str, Any],
        initializers: Dict[str, np.ndarray],
        node: Any,
        parts: List[str],
        bin_file: Any,
    ) -> None:
        """Append NCNN layer-specific parameters and write binary weights."""
        if op_type == "Conv":
            weight_name = node.input[1] if len(node.input) > 1 else None
            bias_name = node.input[2] if len(node.input) > 2 else None
            w = initializers.get(weight_name) if weight_name else None
            if w is not None:
                out_ch, in_ch, kh, kw = w.shape
                groups = self._get_attr_int(attrs, "group", 1)
                pads = self._get_attr_ints(attrs, "pads", [0, 0, 0, 0])
                strides = self._get_attr_ints(attrs, "strides", [1, 1])
                dilations = self._get_attr_ints(attrs, "dilations", [1, 1])
                parts.append(f"0={out_ch}")
                parts.append(f"1={kw}")
                parts.append(f"11={kh}")
                parts.append(f"2={dilations[1]}")
                parts.append(f"12={dilations[0]}")
                parts.append(f"3={strides[1]}")
                parts.append(f"13={strides[0]}")
                parts.append(f"4={pads[1]}")
                parts.append(f"14={pads[0]}")
                parts.append(f"5={1 if bias_name and bias_name in initializers else 0}")
                parts.append(f"6={w.size}")
                parts.append(f"7={groups}")
                data = w.astype(np.float16 if self.fp16 else np.float32).tobytes()
                bin_file.write(data)
                if bias_name and bias_name in initializers:
                    b = initializers[bias_name].astype(np.float32).tobytes()
                    bin_file.write(b)

        elif op_type == "Gemm":
            weight_name = node.input[1] if len(node.input) > 1 else None
            bias_name = node.input[2] if len(node.input) > 2 else None
            w = initializers.get(weight_name) if weight_name else None
            if w is not None:
                out_features = w.shape[0]
                parts.append(f"0={out_features}")
                parts.append(f"1={1 if bias_name and bias_name in initializers else 0}")
                parts.append(f"2={w.size}")
                trans_b = self._get_attr_int(attrs, "transB", 1)
                weight = w.T if not trans_b else w
                bin_file.write(weight.astype(np.float32).tobytes())
                if bias_name and bias_name in initializers:
                    bin_file.write(initializers[bias_name].astype(np.float32).tobytes())

        elif op_type in ("MaxPool", "AveragePool"):
            kernel = self._get_attr_ints(attrs, "kernel_shape", [1, 1])
            strides = self._get_attr_ints(attrs, "strides", [1, 1])
            pads = self._get_attr_ints(attrs, "pads", [0, 0, 0, 0])
            pool_type = 0 if op_type == "MaxPool" else 1
            parts.append(f"0={pool_type}")
            parts.append(f"1={kernel[0]}")
            parts.append(f"2={strides[0]}")
            parts.append(f"3={pads[0]}")

        elif op_type == "GlobalAveragePool":
            parts.append("0=1")
            parts.append("4=1")

        elif op_type == "BatchNormalization":
            scale_name = node.input[1] if len(node.input) > 1 else None
            if scale_name and scale_name in initializers:
                channels = initializers[scale_name].shape[0]
                parts.append(f"0={channels}")
                for idx in range(1, min(5, len(node.input))):
                    name = node.input[idx]
                    if name in initializers:
                        bin_file.write(initializers[name].astype(np.float32).tobytes())

        elif op_type == "Reshape":
            shape_name = node.input[1] if len(node.input) > 1 else None
            if shape_name and shape_name in initializers:
                shape = initializers[shape_name].tolist()
                for i, s in enumerate(shape):
                    parts.append(f"{i}={int(s)}")

        elif op_type == "Softmax":
            axis = self._get_attr_int(attrs, "axis", -1)
            parts.append(f"0={axis}")

        elif op_type == "Add":
            parts.append("0=0")

        elif op_type == "Concat":
            axis = self._get_attr_int(attrs, "axis", 1)
            parts.append(f"0={axis - 1}")

    @staticmethod
    def _get_attr_int(attrs: Dict[str, Any], name: str, default: int) -> int:
        if name in attrs:
            return int(attrs[name].i)
        return default

    @staticmethod
    def _get_attr_ints(attrs: Dict[str, Any], name: str, default: List[int]) -> List[int]:
        if name in attrs:
            return list(attrs[name].ints)
        return default

    # ------------------------------------------------------------------
    # Validation & info
    # ------------------------------------------------------------------

    def validate(self, param_path: str) -> Dict[str, Any]:
        """Check that a ``.param`` file is parseable.

        Args:
            param_path: Path to the NCNN param file.

        Returns:
            Dictionary with ``valid``, ``layer_count``, ``blob_count``, and ``layers``.
        """
        result: Dict[str, Any] = {"valid": False, "layer_count": 0, "blob_count": 0, "layers": []}
        try:
            with open(param_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines or lines[0] != "7767517":
                result["error"] = "Missing NCNN magic number"
                return result

            layer_count, blob_count = map(int, lines[1].split())
            result["layer_count"] = layer_count
            result["blob_count"] = blob_count

            for line in lines[2:]:
                tokens = line.split()
                if len(tokens) >= 4:
                    result["layers"].append({"type": tokens[0], "name": tokens[1]})

            result["valid"] = len(result["layers"]) == layer_count
        except Exception as exc:
            result["error"] = str(exc)
        return result

    def get_model_info(self, param_path: str) -> Dict[str, Any]:
        """Retrieve summary information from NCNN param/bin files.

        Args:
            param_path: Path to the ``.param`` file.

        Returns:
            Dictionary with layer statistics and file sizes.
        """
        validation = self.validate(param_path)
        bin_path = str(Path(param_path).with_suffix(".bin"))
        bin_size = Path(bin_path).stat().st_size if Path(bin_path).exists() else 0

        layer_types: Dict[str, int] = {}
        for layer in validation.get("layers", []):
            t = layer["type"]
            layer_types[t] = layer_types.get(t, 0) + 1

        return {
            "param_file": param_path,
            "bin_file": bin_path,
            "param_size_kb": Path(param_path).stat().st_size / 1024,
            "bin_size_mb": bin_size / (1024 * 1024),
            "layer_count": validation["layer_count"],
            "blob_count": validation["blob_count"],
            "layer_types": layer_types,
            "valid": validation["valid"],
        }
