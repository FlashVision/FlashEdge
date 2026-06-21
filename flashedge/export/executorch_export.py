"""ExecuTorch export for on-device PyTorch inference.

Exports models to the ExecuTorch ``.pte`` format, with optional delegate
lowering (XNNPACK, CoreML, etc.) and a TorchScript fallback for
environments where ExecuTorch is not installed.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from flashedge.registry import EXPORTERS


@EXPORTERS.register("executorch")
class ExecuTorchExporter:
    """Export PyTorch models to ExecuTorch ``.pte`` programs.

    The exporter first captures the model with ``torch.export``, optionally
    lowers sub-graphs to a hardware delegate, and serialises the result as a
    ``.pte`` file.  When ExecuTorch is unavailable it falls back to
    ``torch.jit.trace``.

    Args:
        delegate_backend: Optional delegate name (``"xnnpack"``, ``"coreml"``,
            ``"qnn"``, etc.).  ``None`` disables delegation.
        optimize: Apply graph-level optimisations before serialisation.
        quantize: Run post-export quantisation through the ExecuTorch
            quantisation pipeline.
    """

    KNOWN_DELEGATES = {"xnnpack", "coreml", "qnn", "vulkan", "mps"}

    def __init__(
        self,
        delegate_backend: Optional[str] = None,
        optimize: bool = True,
        quantize: bool = False,
    ) -> None:
        if delegate_backend is not None and delegate_backend not in self.KNOWN_DELEGATES:
            print(f"  Warning: delegate '{delegate_backend}' not in known set {self.KNOWN_DELEGATES}")
        self.delegate_backend = delegate_backend
        self.optimize = optimize
        self.quantize = quantize

    def export(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> str:
        """Export a PyTorch model to ExecuTorch format.

        Args:
            model: The PyTorch model to export.
            output_path: Destination path for the ``.pte`` file.
            input_shape: Shape of the example input tensor.

        Returns:
            Path to the saved ``.pte`` (or ``.pt`` fallback) file.
        """
        model = model.cpu().eval()
        output_path = str(Path(output_path).with_suffix(".pte"))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            return self._export_executorch(model, output_path, input_shape)
        except Exception as exc:
            print(f"  ExecuTorch export failed ({exc}), falling back to TorchScript")
            return self._export_with_torchscript_fallback(model, output_path, input_shape)

    # ------------------------------------------------------------------
    # ExecuTorch path
    # ------------------------------------------------------------------

    def _export_executorch(self, model: nn.Module, output_path: str, input_shape: Tuple[int, ...]) -> str:
        """Full ExecuTorch export pipeline."""
        from torch.export import export as torch_export

        dummy_input = (torch.randn(*input_shape),)

        print("  Capturing model with torch.export...")
        with torch.no_grad():
            exported_program = torch_export(model, dummy_input)

        if self.optimize:
            exported_program = self._apply_optimizations(exported_program)

        if self.quantize:
            exported_program = self._apply_quantization(exported_program, dummy_input)

        if self.delegate_backend:
            exported_program = self._lower_to_delegate(exported_program, dummy_input)

        try:
            from executorch.exir import to_edge, EdgeCompileConfig

            print("  Converting to EdgeProgram...")
            edge_config = EdgeCompileConfig(_check_ir_validity=True)
            edge_program = to_edge(exported_program, compile_config=edge_config)
            et_program = edge_program.to_executorch()

            with open(output_path, "wb") as f:
                f.write(et_program.buffer)
        except ImportError:
            print("  executorch.exir not available, serialising ExportedProgram directly")
            torch.save(exported_program, output_path)

        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  ExecuTorch model saved: {output_path} ({size_mb:.2f} MB)")
        return output_path

    def _apply_optimizations(self, exported_program: Any) -> Any:
        """Apply graph optimisations to the exported program."""
        try:

            print("  Applying graph optimisations...")
            gm = exported_program.graph_module
            gm.graph.eliminate_dead_code()
            gm.recompile()
        except Exception as exc:
            print(f"  Warning: optimisation pass skipped ({exc})")
        return exported_program

    def _apply_quantization(self, exported_program: Any, example_inputs: Tuple[torch.Tensor, ...]) -> Any:
        """Apply post-export quantisation via ExecuTorch's quantiser APIs."""
        try:
            from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
            from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config

            print("  Applying post-export quantisation...")
            quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
            prepared = prepare_pt2e(exported_program, quantizer)

            with torch.no_grad():
                for _ in range(10):
                    prepared.graph_module(*example_inputs)

            exported_program = convert_pt2e(prepared)
        except ImportError:
            print("  Warning: PT2E quantisation APIs not available, skipping")
        except Exception as exc:
            print(f"  Warning: quantisation failed ({exc}), continuing without")
        return exported_program

    def _lower_to_delegate(self, exported_program: Any, example_inputs: Tuple[torch.Tensor, ...]) -> Any:
        """Lower compatible sub-graphs to the specified hardware delegate."""
        backend = self.delegate_backend
        print(f"  Lowering to {backend} delegate...")

        try:
            if backend == "xnnpack":
                from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
                from executorch.exir import to_edge

                edge = to_edge(exported_program)
                edge = edge.to_backend(XnnpackPartitioner())
                return edge
            elif backend == "coreml":
                from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner

                edge = self._to_edge(exported_program)
                edge = edge.to_backend(CoreMLPartitioner())
                return edge
            else:
                print(f"  Warning: delegate '{backend}' not yet integrated, skipping lowering")
        except ImportError:
            print(f"  Warning: {backend} delegate package not installed, skipping")
        except Exception as exc:
            print(f"  Warning: delegate lowering failed ({exc}), continuing without")

        return exported_program

    @staticmethod
    def _to_edge(exported_program: Any) -> Any:
        from executorch.exir import to_edge

        return to_edge(exported_program)

    # ------------------------------------------------------------------
    # TorchScript fallback
    # ------------------------------------------------------------------

    def _export_with_torchscript_fallback(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...],
    ) -> str:
        """Fallback export using ``torch.jit.trace`` when ExecuTorch is unavailable.

        Args:
            model: The PyTorch model.
            output_path: Destination file path.
            input_shape: Shape for the trace input.

        Returns:
            Path to the saved TorchScript model.
        """
        fallback_path = str(Path(output_path).with_suffix(".pt"))
        model = copy.deepcopy(model).cpu().eval()
        dummy_input = torch.randn(*input_shape)

        print("  Tracing model with torch.jit.trace...")
        with torch.no_grad():
            traced = torch.jit.trace(model, dummy_input)

        if self.optimize:
            traced = torch.jit.optimize_for_inference(traced)

        traced.save(fallback_path)

        size_mb = Path(fallback_path).stat().st_size / (1024 * 1024)
        print(f"  TorchScript fallback saved: {fallback_path} ({size_mb:.2f} MB)")
        return fallback_path

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        pte_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> Dict[str, Any]:
        """Validate an exported model file.

        Args:
            pte_path: Path to the ``.pte`` or ``.pt`` file.
            input_shape: Shape of the test input.

        Returns:
            Dictionary with validation results including ``valid``,
            ``format``, ``file_size_mb``, and ``inference_ok``.
        """
        result: Dict[str, Any] = {
            "valid": False,
            "file_size_mb": 0.0,
            "format": "unknown",
            "inference_ok": False,
        }

        path = Path(pte_path)
        if not path.exists():
            result["error"] = "File not found"
            return result

        result["file_size_mb"] = path.stat().st_size / (1024 * 1024)

        if path.suffix == ".pt":
            result["format"] = "torchscript"
            try:
                loaded = torch.jit.load(str(path))
                dummy = torch.randn(*input_shape)
                with torch.no_grad():
                    output = loaded(dummy)
                result["valid"] = True
                result["inference_ok"] = True
                result["output_shape"] = list(output.shape)
            except Exception as exc:
                result["error"] = str(exc)
        elif path.suffix == ".pte":
            result["format"] = "executorch"
            try:
                from executorch.runtime import Runtime

                runtime = Runtime.get()
                program = runtime.load_program(str(path))
                method = program.load_method("forward")
                dummy = torch.randn(*input_shape)
                outputs = method.execute([dummy])
                result["valid"] = True
                result["inference_ok"] = True
                result["output_shape"] = list(outputs[0].shape)
            except ImportError:
                result["valid"] = path.stat().st_size > 0
                result["note"] = "ExecuTorch runtime not available for inference test"
            except Exception as exc:
                result["error"] = str(exc)
        else:
            result["error"] = f"Unsupported file extension: {path.suffix}"

        return result
