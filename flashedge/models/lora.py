"""Low-Rank Adaptation (LoRA) for efficient edge model fine-tuning."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer that wraps a Linear or Conv2d module.

    Args:
        original: The original layer to adapt.
        rank: LoRA rank (lower = fewer params, less capacity).
        alpha: LoRA scaling factor.
        dropout: Dropout rate applied to LoRA path.
    """

    def __init__(
        self,
        original: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        original.requires_grad_(False)

        if isinstance(original, nn.Linear):
            in_features = original.in_features
            out_features = original.out_features
            self.lora_A = nn.Parameter(torch.empty(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        elif isinstance(original, nn.Conv2d):
            in_channels = original.in_channels
            out_channels = original.out_channels
            kernel_size = original.kernel_size[0]
            self.lora_A = nn.Parameter(torch.empty(rank, in_channels * kernel_size * kernel_size))
            self.lora_B = nn.Parameter(torch.zeros(out_channels, rank))
        else:
            raise TypeError(f"LoRA only supports Linear and Conv2d, got {type(original)}")

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)

        if isinstance(self.original, nn.Linear):
            lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        elif isinstance(self.original, nn.Conv2d):
            b, c, h, w = x.shape
            k = self.original.kernel_size[0]
            patches = F.unfold(x, kernel_size=k, padding=self.original.padding[0], stride=self.original.stride[0])
            lora_out = (self.dropout(patches.transpose(1, 2)) @ self.lora_A.T @ self.lora_B.T).transpose(1, 2)
            out_h = (h + 2 * self.original.padding[0] - k) // self.original.stride[0] + 1
            out_w = (w + 2 * self.original.padding[1] - k) // self.original.stride[1] + 1
            lora_out = lora_out.reshape(b, -1, out_h, out_w)
        else:
            return base_out

        return base_out + lora_out * self.scaling


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.0,
) -> nn.Module:
    """Apply LoRA to specified modules in a model.

    Args:
        model: The model to adapt.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        target_modules: List of module name patterns to apply LoRA to.
                       If None, applies to all Linear layers.
        dropout: Dropout rate for LoRA path.

    Returns:
        Model with LoRA layers applied.
    """
    replacements: Dict[str, LoRALayer] = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if target_modules is None:
                if isinstance(module, nn.Linear):
                    replacements[name] = LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout)
            else:
                if any(pattern in name for pattern in target_modules):
                    replacements[name] = LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout)

    for name, lora_layer in replacements.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_layer)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA applied — trainable: {trainable:,} / {total:,} ({trainable / total:.2%})")

    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into the base model for deployment.

    After merging, LoRA layers are replaced with standard layers
    containing the merged weights, with no additional inference cost.

    Args:
        model: Model with LoRA layers.

    Returns:
        Model with LoRA weights merged into base layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            original = module.original

            if isinstance(original, nn.Linear):
                delta = (module.lora_B @ module.lora_A) * module.scaling
                original.weight.data += delta
            elif isinstance(original, nn.Conv2d):
                delta = (module.lora_B @ module.lora_A) * module.scaling
                k = original.kernel_size[0]
                delta = delta.reshape(
                    original.out_channels,
                    original.in_channels,
                    k,
                    k,
                )
                original.weight.data += delta

            original.requires_grad_(True)

            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], original)

    return model
