"""TorchScript tracing and scripting for mobile/C++ deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from flashedge.registry import EXPORTERS


@EXPORTERS.register("torchscript")
class TorchScriptExporter:
    """Export PyTorch models to TorchScript via tracing or scripting.

    Args:
        method: Export method — "trace" or "script".
        optimize_for_mobile: Whether to apply mobile optimization passes.
    """

    def __init__(
        self,
        method: str = "trace",
        optimize_for_mobile: bool = False,
    ) -> None:
        if method not in ("trace", "script"):
            raise ValueError(f"method must be 'trace' or 'script', got '{method}'")
        self.method = method
        self.optimize_for_mobile = optimize_for_mobile

    def export(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> str:
        """Export a PyTorch model to TorchScript format.

        Args:
            model: The PyTorch model to export.
            output_path: Path to save the TorchScript model (.pt).
            input_shape: Shape of the input tensor (used for tracing).

        Returns:
            Path to the exported TorchScript model.
        """
        model = model.cpu().eval()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.method == "trace":
            dummy_input = torch.randn(*input_shape)
            with torch.no_grad():
                scripted = torch.jit.trace(model, dummy_input)
        else:
            scripted = torch.jit.script(model)

        if self.optimize_for_mobile:
            scripted = self._optimize_for_mobile(scripted)

        scripted.save(str(output_path))

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  TorchScript model saved: {output_path} ({size_mb:.2f} MB)")

        return str(output_path)

    def _optimize_for_mobile(self, scripted_model: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """Apply mobile optimization passes."""
        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile

            return optimize_for_mobile(scripted_model)
        except ImportError:
            print("  Warning: Mobile optimization not available in this PyTorch build")
            return scripted_model

    def validate(
        self,
        ts_path: str,
        model: Optional[nn.Module] = None,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        atol: float = 1e-5,
    ) -> Dict[str, Any]:
        """Validate a TorchScript model."""
        loaded = torch.jit.load(ts_path, map_location="cpu")
        loaded.eval()

        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            ts_output = loaded(dummy)

        result: Dict[str, Any] = {
            "load_ok": True,
            "inference_ok": True,
            "output_shape": list(ts_output.shape),
        }

        if model is not None:
            model = model.cpu().eval()
            with torch.no_grad():
                pt_output = model(dummy)
            max_diff = float((pt_output - ts_output).abs().max())
            result["max_output_diff"] = max_diff
            result["outputs_match"] = max_diff < atol

        return result
