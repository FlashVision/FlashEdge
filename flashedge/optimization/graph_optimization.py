"""Operator fusion and graph-level optimisation passes.

Provides Conv-BN fusion, Conv-ReLU fusion, constant folding, and dead-code
elimination for PyTorch ``nn.Module`` graphs, plus an ONNX-level
optimisation path for exported models.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from flashedge.registry import OPTIMIZERS


class _ConvReLU(nn.Module):
    """Fused Conv2d + ReLU executed as a single module."""

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x))


@OPTIMIZERS.register("graph_optimizer")
class GraphOptimizer:
    """Apply graph-level optimisations to a PyTorch model.

    Each optimisation pass can be individually toggled.  The passes are
    applied in a fixed, dependency-aware order: BN fusion → ReLU fusion →
    constant folding → dead-code elimination.

    Args:
        fuse_conv_bn: Fold BatchNorm parameters into preceding Conv2d layers.
        fuse_conv_relu: Replace Conv2d + ReLU pairs with a fused module.
        constant_folding: Trace the model and fold constant sub-graphs.
        dead_code_elimination: Remove unused parameters and buffers.
    """

    def __init__(
        self,
        fuse_conv_bn: bool = True,
        fuse_conv_relu: bool = True,
        constant_folding: bool = True,
        dead_code_elimination: bool = True,
    ) -> None:
        self.fuse_conv_bn = fuse_conv_bn
        self.fuse_conv_relu = fuse_conv_relu
        self.constant_folding = constant_folding
        self.dead_code_elimination = dead_code_elimination

    def optimize(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> nn.Module:
        """Apply all enabled optimisation passes sequentially.

        Args:
            model: The PyTorch model to optimise.
            input_shape: Shape for the trace dummy input (constant folding).

        Returns:
            Optimised model.
        """
        model = copy.deepcopy(model).cpu().eval()
        before_stats = self._model_stats(model)

        if self.fuse_conv_bn:
            model = self._fuse_conv_bn(model)

        if self.fuse_conv_relu:
            model = self._fuse_conv_relu(model)

        if self.constant_folding:
            model = self._constant_folding(model, input_shape)

        if self.dead_code_elimination:
            model = self._dead_code_elimination(model)

        after_stats = self._model_stats(model)
        fusions = self._count_fusions(before_stats, after_stats)
        print(f"  Graph optimisation complete: {fusions.get('summary', 'no changes')}")

        return model

    # ------------------------------------------------------------------
    # Conv + BN fusion
    # ------------------------------------------------------------------

    def _fuse_conv_bn(self, model: nn.Module) -> nn.Module:
        """Fold BatchNorm into the preceding Conv2d using PyTorch's built-in utility.

        Args:
            model: Model in eval mode.

        Returns:
            Model with fused Conv-BN pairs.
        """
        fused_count = 0
        children = list(model.named_children())
        for i, (name, module) in enumerate(children):
            if isinstance(module, nn.Sequential):
                setattr(model, name, self._fuse_conv_bn(module))
                continue

            if isinstance(module, (nn.Conv2d,)) and i + 1 < len(children):
                next_name, next_module = children[i + 1]
                if isinstance(next_module, nn.BatchNorm2d) and next_module.num_features == module.out_channels:
                    fused = torch.nn.utils.fuse_conv_bn_eval(module, next_module)
                    setattr(model, name, fused)
                    setattr(model, next_name, nn.Identity())
                    fused_count += 1

            if len(list(module.children())) > 0:
                setattr(model, name, self._fuse_conv_bn(module))

        if fused_count:
            print(f"  Fused {fused_count} Conv+BN pairs")

        return model

    # ------------------------------------------------------------------
    # Conv + ReLU fusion
    # ------------------------------------------------------------------

    def _fuse_conv_relu(self, model: nn.Module) -> nn.Module:
        """Replace Conv2d + ReLU sequences with a fused ``_ConvReLU`` module.

        Args:
            model: Model (possibly already BN-fused).

        Returns:
            Model with fused Conv-ReLU modules.
        """
        fused_count = 0
        children = list(model.named_children())
        skip_next = False

        for i, (name, module) in enumerate(children):
            if skip_next:
                skip_next = False
                continue

            if isinstance(module, nn.Sequential):
                setattr(model, name, self._fuse_conv_relu(module))
                continue

            if isinstance(module, nn.Conv2d) and i + 1 < len(children):
                next_name, next_module = children[i + 1]
                if isinstance(next_module, (nn.ReLU, nn.ReLU6)):
                    setattr(model, name, _ConvReLU(module))
                    setattr(model, next_name, nn.Identity())
                    fused_count += 1
                    skip_next = True
                    continue

            if len(list(module.children())) > 0:
                setattr(model, name, self._fuse_conv_relu(module))

        if fused_count:
            print(f"  Fused {fused_count} Conv+ReLU pairs")

        return model

    # ------------------------------------------------------------------
    # Constant folding
    # ------------------------------------------------------------------

    def _constant_folding(self, model: nn.Module, input_shape: Tuple[int, ...]) -> nn.Module:
        """Trace the model and apply constant folding via ``torch.jit``.

        Args:
            model: The eval-mode model.
            input_shape: Shape for the trace input.

        Returns:
            The model, potentially with folded constants.
        """
        try:
            dummy = torch.randn(*input_shape)
            with torch.no_grad():
                traced = torch.jit.trace(model, dummy)
            traced = torch.jit.freeze(traced)

            with torch.no_grad():
                traced(dummy)

            print("  Constant folding applied via torch.jit.freeze")
            return traced
        except Exception as exc:
            print(f"  Warning: constant folding skipped ({exc})")
            return model

    # ------------------------------------------------------------------
    # Dead-code elimination
    # ------------------------------------------------------------------

    def _dead_code_elimination(self, model: nn.Module) -> nn.Module:
        """Remove ``nn.Identity`` modules and unused parameters / buffers.

        Args:
            model: The model after fusion passes.

        Returns:
            Cleaned-up model.
        """
        removed = 0

        if isinstance(model, torch.jit.ScriptModule):
            return model

        identity_names: List[str] = []
        for name, module in model.named_children():
            if isinstance(module, nn.Identity):
                identity_names.append(name)
            elif len(list(module.children())) > 0:
                setattr(model, name, self._dead_code_elimination(module))

        for name in identity_names:
            delattr(model, name)
            removed += 1

        unused_buffers: List[str] = []
        for name, buf in model.named_buffers(recurse=False):
            if buf.numel() == 0:
                unused_buffers.append(name)

        for name in unused_buffers:
            delattr(model, name)
            removed += 1

        if removed:
            print(f"  Removed {removed} dead nodes / empty buffers")

        return model

    # ------------------------------------------------------------------
    # Statistics & reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _model_stats(model: nn.Module) -> Dict[str, int]:
        """Count modules by type."""
        counts: Dict[str, int] = {}
        for _, module in model.named_modules():
            key = module.__class__.__name__
            counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _count_fusions(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, Any]:
        """Compare module counts before and after optimisation.

        Args:
            before: Module-type counts before optimisation.
            after: Module-type counts after optimisation.

        Returns:
            Dictionary with per-type deltas and a human-readable summary.
        """
        all_keys = set(before) | set(after)
        deltas: Dict[str, int] = {}
        parts: List[str] = []

        for key in sorted(all_keys):
            b = before.get(key, 0)
            a = after.get(key, 0)
            if b != a:
                deltas[key] = a - b
                parts.append(f"{key}: {b}→{a}")

        return {
            "deltas": deltas,
            "summary": ", ".join(parts) if parts else "no changes",
        }

    def analyze(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> Dict[str, Any]:
        """Analyse optimisation opportunities without modifying the model.

        Args:
            model: The PyTorch model.
            input_shape: Dummy input shape.

        Returns:
            Dictionary listing fusible pairs and removable nodes.
        """
        model = copy.deepcopy(model).cpu().eval()

        conv_bn_pairs: List[Tuple[str, str]] = []
        conv_relu_pairs: List[Tuple[str, str]] = []
        identity_nodes: List[str] = []

        self._find_fusion_candidates(model, "", conv_bn_pairs, conv_relu_pairs, identity_nodes)

        total_params = sum(p.numel() for p in model.parameters())

        return {
            "conv_bn_pairs": conv_bn_pairs,
            "conv_relu_pairs": conv_relu_pairs,
            "identity_nodes": identity_nodes,
            "fusible_conv_bn": len(conv_bn_pairs),
            "fusible_conv_relu": len(conv_relu_pairs),
            "removable_identity": len(identity_nodes),
            "total_params": total_params,
        }

    def _find_fusion_candidates(
        self,
        model: nn.Module,
        prefix: str,
        conv_bn: List[Tuple[str, str]],
        conv_relu: List[Tuple[str, str]],
        identities: List[str],
    ) -> None:
        children = list(model.named_children())
        for i, (name, module) in enumerate(children):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(module, nn.Identity):
                identities.append(full_name)

            if isinstance(module, nn.Conv2d) and i + 1 < len(children):
                _, next_mod = children[i + 1]
                if isinstance(next_mod, nn.BatchNorm2d):
                    conv_bn.append((full_name, f"{prefix}.{children[i + 1][0]}" if prefix else children[i + 1][0]))
                elif isinstance(next_mod, (nn.ReLU, nn.ReLU6)):
                    conv_relu.append((full_name, f"{prefix}.{children[i + 1][0]}" if prefix else children[i + 1][0]))

            if len(list(module.children())) > 0:
                self._find_fusion_candidates(module, full_name, conv_bn, conv_relu, identities)

    # ------------------------------------------------------------------
    # ONNX-level optimisation
    # ------------------------------------------------------------------

    def optimize_onnx(self, onnx_path: str, output_path: str) -> str:
        """Apply graph optimisations to an ONNX model.

        Uses ONNX Runtime's graph optimisation session to fuse ops and
        eliminate redundancies.

        Args:
            onnx_path: Path to the input ONNX model.
            output_path: Path to save the optimised model.

        Returns:
            Path to the optimised model.
        """
        import onnxruntime as ort

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.optimized_model_filepath = output_path

        ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])

        size_before = Path(onnx_path).stat().st_size / (1024 * 1024)
        size_after = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  ONNX graph optimised: {size_before:.2f} → {size_after:.2f} MB")

        return output_path
