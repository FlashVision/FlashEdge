"""FLOPs and MACs counting for model complexity analysis.

Computes Floating Point Operations (FLOPs) and Multiply-Accumulate
operations (MACs) by hooking into each layer's forward pass.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


class FLOPsCounter:
    """Count FLOPs and MACs for a PyTorch model.

    Supports Conv2d, Linear, BatchNorm2d, ReLU, and pooling layers.
    """

    def count(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> Dict[str, Any]:
        """Count total FLOPs and MACs for a model.

        Args:
            model: PyTorch model to analyze.
            input_shape: Shape of the input tensor.

        Returns:
            Dictionary with FLOPs, MACs, and per-layer breakdown.
        """
        model = model.cpu().eval()
        layer_stats: List[Dict[str, Any]] = []
        hooks = []

        def make_hook(name: str, module: nn.Module):
            def hook_fn(mod: nn.Module, inp: Any, out: Any) -> None:
                flops, macs = self._compute_layer_flops(mod, inp, out)
                layer_stats.append(
                    {
                        "name": name,
                        "type": mod.__class__.__name__,
                        "flops": flops,
                        "macs": macs,
                        "output_shape": list(out.shape) if isinstance(out, torch.Tensor) else "N/A",
                    }
                )

            return hook_fn

        for name, module in model.named_modules():
            if isinstance(
                module,
                (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm, nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d),
            ):
                hooks.append(module.register_forward_hook(make_hook(name, module)))

        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            model(dummy)

        for h in hooks:
            h.remove()

        total_flops = sum(s["flops"] for s in layer_stats)
        total_macs = sum(s["macs"] for s in layer_stats)

        return {
            "flops": total_flops,
            "macs": total_macs,
            "flops_str": self._format_count(total_flops),
            "macs_str": self._format_count(total_macs),
            "per_layer": layer_stats,
        }

    def _compute_layer_flops(
        self,
        module: nn.Module,
        inp: Any,
        out: Any,
    ) -> Tuple[int, int]:
        """Compute FLOPs and MACs for a single layer."""
        if isinstance(module, nn.Conv2d):
            return self._conv2d_flops(module, inp, out)
        elif isinstance(module, nn.Linear):
            return self._linear_flops(module, inp, out)
        elif isinstance(module, nn.BatchNorm2d):
            return self._batchnorm_flops(module, inp, out)
        elif isinstance(module, nn.LayerNorm):
            return self._layernorm_flops(module, inp, out)
        elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
            return self._pool_flops(module, inp, out)
        return 0, 0

    def _conv2d_flops(
        self,
        module: nn.Conv2d,
        inp: Any,
        out: Any,
    ) -> Tuple[int, int]:
        """FLOPs for Conv2d: 2 * Cout * Cin * Kh * Kw * Hout * Wout / groups."""
        output = out if isinstance(out, torch.Tensor) else out[0]
        batch, cout, hout, wout = output.shape
        cin = module.in_channels
        kh, kw = module.kernel_size
        groups = module.groups

        macs = (cin // groups) * kh * kw * cout * hout * wout
        flops = 2 * macs

        if module.bias is not None:
            flops += cout * hout * wout

        return flops, macs

    def _linear_flops(
        self,
        module: nn.Linear,
        inp: Any,
        out: Any,
    ) -> Tuple[int, int]:
        """FLOPs for Linear: 2 * in_features * out_features * batch."""
        output = out if isinstance(out, torch.Tensor) else out[0]
        batch_size = output.shape[0]

        macs = module.in_features * module.out_features
        flops = 2 * macs

        if module.bias is not None:
            flops += module.out_features

        total_macs = macs * batch_size
        total_flops = flops * batch_size

        return total_flops, total_macs

    def _batchnorm_flops(
        self,
        module: nn.BatchNorm2d,
        inp: Any,
        out: Any,
    ) -> Tuple[int, int]:
        """FLOPs for BatchNorm2d: 4 * num_features * H * W (mean, var, normalize, affine)."""
        output = out if isinstance(out, torch.Tensor) else out[0]
        elements = output.numel()
        flops = 4 * elements
        return flops, 0

    def _layernorm_flops(
        self,
        module: nn.LayerNorm,
        inp: Any,
        out: Any,
    ) -> Tuple[int, int]:
        """FLOPs for LayerNorm."""
        output = out if isinstance(out, torch.Tensor) else out[0]
        elements = output.numel()
        flops = 5 * elements
        return flops, 0

    def _pool_flops(
        self,
        module: nn.Module,
        inp: Any,
        out: Any,
    ) -> Tuple[int, int]:
        """FLOPs for pooling layers."""
        output = out if isinstance(out, torch.Tensor) else out[0]
        flops = output.numel()
        return flops, 0

    def _format_count(self, count: int) -> str:
        """Format a large number with appropriate suffix."""
        if count >= 1e12:
            return f"{count / 1e12:.2f} T"
        elif count >= 1e9:
            return f"{count / 1e9:.2f} G"
        elif count >= 1e6:
            return f"{count / 1e6:.2f} M"
        elif count >= 1e3:
            return f"{count / 1e3:.2f} K"
        return str(count)

    def summary(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> str:
        """Generate a human-readable FLOPs summary.

        Args:
            model: PyTorch model.
            input_shape: Input tensor shape.

        Returns:
            Formatted string with FLOPs breakdown.
        """
        result = self.count(model, input_shape)
        lines = [
            "FLOPs Summary",
            f"{'=' * 70}",
            f"Total FLOPs: {result['flops_str']}",
            f"Total MACs:  {result['macs_str']}",
            f"{'=' * 70}",
            f"{'Layer':<40} {'Type':<15} {'FLOPs':<15}",
            f"{'-' * 70}",
        ]

        for layer in result["per_layer"]:
            lines.append(f"{layer['name']:<40} {layer['type']:<15} {self._format_count(layer['flops']):<15}")

        return "\n".join(lines)
