"""MobileNetV3 architecture implementations for edge deployment."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashedge.registry import MODELS


class HardSwish(nn.Module):
    """Hard Swish activation — efficient approximation of Swish for mobile."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3.0, inplace=True) / 6.0


class HardSigmoid(nn.Module):
    """Hard Sigmoid activation for mobile."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0, inplace=True) / 6.0


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)
        self.act = nn.ReLU(inplace=True)
        self.gate = HardSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.act(self.fc1(scale))
        scale = self.gate(self.fc2(scale))
        return x * scale


class InvertedResidual(nn.Module):
    """MobileNetV3 inverted residual block with SE and nonlinearity options."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: float,
        use_se: bool = False,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        mid_channels = int(in_channels * expand_ratio)

        act_layer = HardSwish if activation == "hswish" else nn.ReLU

        layers: list = []

        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                act_layer(),
            ])

        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            act_layer(),
        ])

        if use_se:
            layers.append(SqueezeExcite(mid_channels))

        layers.extend([
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


@MODELS.register("mobilenetv3_small")
class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small architecture optimized for edge/mobile.

    Args:
        num_classes: Number of output classes.
        width_mult: Width multiplier for channel counts.
    """

    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0) -> None:
        super().__init__()

        def c(channels: int) -> int:
            return max(int(channels * width_mult), 8)

        self.features = nn.Sequential(
            nn.Conv2d(3, c(16), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(16)),
            HardSwish(),
            InvertedResidual(c(16), c(16), 3, 2, 1.0, use_se=True, activation="relu"),
            InvertedResidual(c(16), c(24), 3, 2, 72.0 / 16, use_se=False, activation="relu"),
            InvertedResidual(c(24), c(24), 3, 1, 88.0 / 24, use_se=False, activation="relu"),
            InvertedResidual(c(24), c(40), 5, 2, 4.0, use_se=True, activation="hswish"),
            InvertedResidual(c(40), c(40), 5, 1, 6.0, use_se=True, activation="hswish"),
            InvertedResidual(c(40), c(40), 5, 1, 6.0, use_se=True, activation="hswish"),
            InvertedResidual(c(40), c(48), 5, 1, 3.0, use_se=True, activation="hswish"),
            InvertedResidual(c(48), c(48), 5, 1, 3.0, use_se=True, activation="hswish"),
            InvertedResidual(c(48), c(96), 5, 2, 6.0, use_se=True, activation="hswish"),
            InvertedResidual(c(96), c(96), 5, 1, 6.0, use_se=True, activation="hswish"),
            InvertedResidual(c(96), c(96), 5, 1, 6.0, use_se=True, activation="hswish"),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(c(96), c(576), 1, bias=False),
            nn.BatchNorm2d(c(576)),
            HardSwish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c(576), c(1024)),
            HardSwish(),
            nn.Dropout(0.2),
            nn.Linear(c(1024), num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


@MODELS.register("mobilenetv3_large")
class MobileNetV3Large(nn.Module):
    """MobileNetV3-Large architecture.

    Args:
        num_classes: Number of output classes.
        width_mult: Width multiplier.
    """

    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0) -> None:
        super().__init__()

        def c(channels: int) -> int:
            return max(int(channels * width_mult), 8)

        self.features = nn.Sequential(
            nn.Conv2d(3, c(16), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(16)),
            HardSwish(),
            InvertedResidual(c(16), c(16), 3, 1, 1.0, use_se=False, activation="relu"),
            InvertedResidual(c(16), c(24), 3, 2, 4.0, use_se=False, activation="relu"),
            InvertedResidual(c(24), c(24), 3, 1, 3.0, use_se=False, activation="relu"),
            InvertedResidual(c(24), c(40), 5, 2, 3.0, use_se=True, activation="relu"),
            InvertedResidual(c(40), c(40), 5, 1, 3.0, use_se=True, activation="relu"),
            InvertedResidual(c(40), c(40), 5, 1, 3.0, use_se=True, activation="relu"),
            InvertedResidual(c(40), c(80), 3, 2, 6.0, use_se=False, activation="hswish"),
            InvertedResidual(c(80), c(80), 3, 1, 2.5, use_se=False, activation="hswish"),
            InvertedResidual(c(80), c(80), 3, 1, 2.3, use_se=False, activation="hswish"),
            InvertedResidual(c(80), c(80), 3, 1, 2.3, use_se=False, activation="hswish"),
            InvertedResidual(c(80), c(112), 3, 1, 6.0, use_se=True, activation="hswish"),
            InvertedResidual(c(112), c(112), 3, 1, 6.0, use_se=True, activation="hswish"),
            InvertedResidual(c(112), c(160), 5, 2, 6.0, use_se=True, activation="hswish"),
            InvertedResidual(c(160), c(160), 5, 1, 6.0, use_se=True, activation="hswish"),
            InvertedResidual(c(160), c(160), 5, 1, 6.0, use_se=True, activation="hswish"),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(c(160), c(960), 1, bias=False),
            nn.BatchNorm2d(c(960)),
            HardSwish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c(960), c(1280)),
            HardSwish(),
            nn.Dropout(0.2),
            nn.Linear(c(1280), num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
