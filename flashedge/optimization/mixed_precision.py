"""Mixed-precision per-layer quantisation for edge deployment.

Performs sensitivity analysis to assign each layer its optimal bitwidth
from a candidate set (e.g. 4, 8, 16, 32), balancing model accuracy against
size.  A greedy Pareto search ensures the overall quality degradation stays
within a user-specified budget.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from flashedge.registry import OPTIMIZERS


@OPTIMIZERS.register("mixed_precision")
class MixedPrecisionOptimizer:
    """Assign per-layer bitwidths to minimise model size under a quality budget.

    The optimiser first profiles each quantisable layer's sensitivity by
    measuring output degradation at every candidate bitwidth, then runs a
    greedy Pareto search to find the cheapest configuration that stays
    within ``target_metric_drop``.

    Args:
        target_metric_drop: Maximum tolerable increase in output MSE
            (relative to the FP32 baseline).
        search_method: Search strategy — ``"sensitivity"`` (greedy) or
            ``"exhaustive"`` (for very small models).
        candidate_bits: Tuple of bitwidths to consider per layer.
    """

    def __init__(
        self,
        target_metric_drop: float = 0.01,
        search_method: str = "sensitivity",
        candidate_bits: Tuple[int, ...] = (4, 8, 16, 32),
    ) -> None:
        self.target_metric_drop = target_metric_drop
        self.search_method = search_method
        self.candidate_bits = sorted(candidate_bits)

    def optimize(
        self,
        model: nn.Module,
        calibration_loader: Optional[Any] = None,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> nn.Module:
        """Run mixed-precision optimisation and return the compressed model.

        Args:
            model: The FP32 PyTorch model.
            calibration_loader: Optional data loader for sensitivity analysis.
            input_shape: Shape for synthetic calibration inputs.

        Returns:
            Model with per-layer quantised weights (dequantised to float).
        """
        model = copy.deepcopy(model).cpu().eval()

        sensitivities = self._analyze_sensitivity(model, calibration_loader, input_shape)
        model_size = self._model_size_bytes(model)
        layer_config = self._pareto_search(sensitivities, model_size, self.target_metric_drop)
        model = self._apply_precision(model, layer_config)

        total_layers = len(layer_config)
        bit_counts = {}
        for bits in layer_config.values():
            bit_counts[bits] = bit_counts.get(bits, 0) + 1
        summary = ", ".join(f"{b}b: {c}" for b, c in sorted(bit_counts.items()))
        print(f"  Mixed-precision complete: {total_layers} layers ({summary})")

        return model

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def _analyze_sensitivity(
        self,
        model: nn.Module,
        calibration_loader: Optional[Any],
        input_shape: Tuple[int, ...],
    ) -> Dict[str, Dict[int, float]]:
        """Measure per-layer output degradation at each candidate bitwidth.

        For every quantisable layer the method quantises *only* that layer,
        runs inference, and records the MSE between quantised and FP32
        outputs.

        Args:
            model: The FP32 reference model.
            calibration_loader: Optional calibration data.
            input_shape: Shape for synthetic inputs.

        Returns:
            ``{layer_name: {bits: mse_score, ...}, ...}``.
        """
        print("  Analysing per-layer sensitivity...")

        inputs = self._get_calibration_inputs(calibration_loader, input_shape, num_batches=8)

        with torch.no_grad():
            ref_outputs = [model(x) for x in inputs]

        quantizable_layers: Dict[str, nn.Module] = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                quantizable_layers[name] = module

        sensitivities: Dict[str, Dict[int, float]] = {}

        for layer_name, layer_module in quantizable_layers.items():
            sensitivities[layer_name] = {}
            original_weight = layer_module.weight.data.clone()

            for bits in self.candidate_bits:
                if bits == 32:
                    sensitivities[layer_name][32] = 0.0
                    continue

                layer_module.weight.data = self._fake_quantize(original_weight, bits)

                mse_total = 0.0
                with torch.no_grad():
                    for i, x in enumerate(inputs):
                        out = model(x)
                        mse_total += torch.nn.functional.mse_loss(out, ref_outputs[i]).item()

                sensitivities[layer_name][bits] = mse_total / len(inputs)
                layer_module.weight.data = original_weight.clone()

        return sensitivities

    # ------------------------------------------------------------------
    # Pareto search
    # ------------------------------------------------------------------

    def _pareto_search(
        self,
        sensitivities: Dict[str, Dict[int, float]],
        model_size: int,
        target_drop: float,
    ) -> Dict[str, int]:
        """Greedy Pareto-optimal bitwidth assignment.

        Starts by assigning every layer the lowest candidate bitwidth, then
        iteratively upgrades the most sensitive layer until the aggregate
        degradation falls within ``target_drop``.

        Args:
            sensitivities: Per-layer sensitivity mapping.
            model_size: Original model size in bytes.
            target_drop: Maximum tolerable MSE increase.

        Returns:
            ``{layer_name: assigned_bits, ...}``.
        """
        min_bits = self.candidate_bits[0]
        config: Dict[str, int] = {name: min_bits for name in sensitivities}

        def _total_degradation() -> float:
            return sum(sensitivities[n].get(config[n], 0.0) for n in config)

        sorted_bits = sorted(self.candidate_bits)

        while _total_degradation() > target_drop:
            worst_layer: Optional[str] = None
            worst_score = -1.0

            for name in config:
                current_bits = config[name]
                score = sensitivities[name].get(current_bits, 0.0)
                if score > worst_score and current_bits < sorted_bits[-1]:
                    worst_score = score
                    worst_layer = name

            if worst_layer is None:
                break

            current_idx = sorted_bits.index(config[worst_layer])
            if current_idx + 1 < len(sorted_bits):
                config[worst_layer] = sorted_bits[current_idx + 1]
            else:
                break

        return config

    # ------------------------------------------------------------------
    # Apply precision
    # ------------------------------------------------------------------

    def _apply_precision(self, model: nn.Module, layer_config: Dict[str, int]) -> nn.Module:
        """Apply per-layer fake quantisation according to the configuration.

        Args:
            model: The model to quantise in-place.
            layer_config: ``{layer_name: bitwidth}`` mapping.

        Returns:
            The model with quantised weights.
        """
        for name, module in model.named_modules():
            if name not in layer_config:
                continue
            bits = layer_config[name]
            if bits >= 32 or not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            module.weight.data = self._fake_quantize(module.weight.data, bits)

        return model

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_precision_report(self, model: nn.Module) -> Dict[str, Any]:
        """Generate a summary of quantisable layers and estimated compression.

        Args:
            model: The PyTorch model.

        Returns:
            Dictionary with per-layer info and overall compression estimate.
        """
        layers: List[Dict[str, Any]] = []
        total_params = 0
        total_size = 0

        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            numel = module.weight.numel()
            size = numel * module.weight.element_size()
            total_params += numel
            total_size += size
            layers.append(
                {
                    "name": name,
                    "type": module.__class__.__name__,
                    "params": numel,
                    "size_kb": size / 1024,
                    "candidate_bits": list(self.candidate_bits),
                }
            )

        compressed = {}
        for bits in self.candidate_bits:
            compressed[f"{bits}bit_size_mb"] = total_size * (bits / 32) / (1024 * 1024)

        return {
            "total_params": total_params,
            "original_size_mb": total_size / (1024 * 1024),
            "layers": layers,
            "estimated_sizes": compressed,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _fake_quantize(weight: torch.Tensor, bits: int) -> torch.Tensor:
        """Simulate quantisation at the given bitwidth via quantise-dequantise.

        Args:
            weight: Original FP32 weight tensor.
            bits: Target bitwidth (e.g. 4, 8, 16).

        Returns:
            Weight tensor after fake-quantisation round-trip.
        """
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1
        abs_max = weight.abs().amax().clamp(min=1e-8)
        scale = abs_max / qmax
        quantized = (weight / scale).round().clamp(qmin, qmax)
        return quantized * scale

    @staticmethod
    def _model_size_bytes(model: nn.Module) -> int:
        return sum(p.numel() * p.element_size() for p in model.parameters())

    @staticmethod
    def _get_calibration_inputs(
        loader: Optional[Any],
        input_shape: Tuple[int, ...],
        num_batches: int = 8,
    ) -> List[torch.Tensor]:
        inputs: List[torch.Tensor] = []
        if loader is not None:
            for i, batch in enumerate(loader):
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                inputs.append(batch.cpu())
                if i + 1 >= num_batches:
                    break
        else:
            for _ in range(num_batches):
                inputs.append(torch.randn(*input_shape))
        return inputs
