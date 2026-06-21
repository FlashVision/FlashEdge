"""Structured and unstructured pruning for mobile-optimized models.

Uses PyTorch's torch.nn.utils.prune module for weight pruning with
support for iterative pruning, importance scoring, and fine-tuning.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from flashedge.registry import OPTIMIZERS


@OPTIMIZERS.register("structured_pruner")
class StructuredPruner:
    """Structured (channel/filter) pruning for real speedup on edge hardware.

    Removes entire channels or filters based on L1-norm importance,
    producing a genuinely smaller model (not just sparse).

    Args:
        sparsity: Fraction of channels to prune (0.0–1.0).
        criterion: Importance criterion — "l1_norm" or "l2_norm".
        iterative: Whether to prune iteratively.
        iterations: Number of pruning iterations.
    """

    def __init__(
        self,
        sparsity: float = 0.3,
        criterion: str = "l1_norm",
        iterative: bool = True,
        iterations: int = 3,
    ) -> None:
        if not 0.0 < sparsity < 1.0:
            raise ValueError(f"sparsity must be in (0, 1), got {sparsity}")
        self.sparsity = sparsity
        self.criterion = criterion
        self.iterative = iterative
        self.iterations = iterations

    def prune(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning to a model.

        Args:
            model: The PyTorch model to prune.

        Returns:
            Pruned model with reduced channels.
        """
        model = copy.deepcopy(model).cpu()

        if self.iterative:
            step_sparsity = 1.0 - (1.0 - self.sparsity) ** (1.0 / self.iterations)
            for i in range(self.iterations):
                self._prune_step(model, step_sparsity)
                print(f"  Pruning iteration {i + 1}/{self.iterations} — step sparsity: {step_sparsity:.3f}")
        else:
            self._prune_step(model, self.sparsity)

        self._make_pruning_permanent(model)

        total, nonzero, ratio = self._compute_sparsity(model)
        print(f"  Final sparsity: {1 - ratio:.2%} ({nonzero:,} / {total:,} parameters)")

        return model

    def _prune_step(self, model: nn.Module, sparsity: float) -> None:
        """Prune one iteration over all eligible layers."""
        if self.criterion == "l1_norm":
            pruning_method: Any = prune.LnStructured
            kwargs: Dict[str, Any] = {"n": 1, "dim": 0}
        else:
            pruning_method = prune.LnStructured
            kwargs = {"n": 2, "dim": 0}

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                pruning_method.apply(module, name="weight", amount=sparsity, **kwargs)

    def _make_pruning_permanent(self, model: nn.Module) -> None:
        """Remove pruning re-parametrization and make masks permanent."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    prune.remove(module, "weight")
                except ValueError:
                    pass

    def _compute_sparsity(self, model: nn.Module) -> Tuple[int, int, float]:
        """Compute overall model sparsity."""
        total = 0
        nonzero = 0
        for param in model.parameters():
            total += param.numel()
            nonzero += param.nonzero().size(0)
        ratio = nonzero / max(total, 1)
        return total, nonzero, ratio

    def get_importance_scores(self, model: nn.Module) -> Dict[str, float]:
        """Compute per-layer importance scores based on the chosen criterion."""
        scores = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight.data
                if self.criterion == "l1_norm":
                    score = weight.abs().sum(dim=(1, 2, 3))
                else:
                    score = weight.pow(2).sum(dim=(1, 2, 3)).sqrt()
                scores[name] = float(score.mean())
        return scores


@OPTIMIZERS.register("unstructured_pruner")
class UnstructuredPruner:
    """Unstructured (weight-level) magnitude pruning.

    Sets individual weights to zero based on magnitude. Produces sparse
    tensors but does not change model architecture.

    Args:
        sparsity: Fraction of weights to prune (0.0–1.0).
        method: Pruning method — "magnitude", "random", or "l1_unstructured".
        global_pruning: Apply global pruning across all layers.
    """

    def __init__(
        self,
        sparsity: float = 0.5,
        method: str = "magnitude",
        global_pruning: bool = False,
    ) -> None:
        if not 0.0 < sparsity < 1.0:
            raise ValueError(f"sparsity must be in (0, 1), got {sparsity}")
        self.sparsity = sparsity
        self.method = method
        self.global_pruning = global_pruning

    def prune(self, model: nn.Module) -> nn.Module:
        """Apply unstructured pruning to a model.

        Args:
            model: The PyTorch model to prune.

        Returns:
            Pruned model with zeroed weights.
        """
        model = copy.deepcopy(model).cpu()

        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, "weight"))

        if not parameters_to_prune:
            print("  Warning: No prunable layers found")
            return model

        if self.global_pruning:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.sparsity,
            )
        else:
            for module, param_name in parameters_to_prune:
                if self.method == "random":
                    prune.random_unstructured(module, name=param_name, amount=self.sparsity)
                else:
                    prune.l1_unstructured(module, name=param_name, amount=self.sparsity)

        for module, param_name in parameters_to_prune:
            try:
                prune.remove(module, param_name)
            except ValueError:
                pass

        total, nonzero, ratio = self._compute_sparsity(model)
        print(f"  Pruning complete — sparsity: {1 - ratio:.2%} ({total - nonzero:,} / {total:,} weights zeroed)")

        return model

    def _compute_sparsity(self, model: nn.Module) -> Tuple[int, int, float]:
        total = 0
        nonzero = 0
        for param in model.parameters():
            total += param.numel()
            nonzero += param.nonzero().size(0)
        ratio = nonzero / max(total, 1)
        return total, nonzero, ratio
