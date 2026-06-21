"""EfficientNet-Lite architecture for edge deployment.

A mobile-friendly variant of EfficientNet that replaces squeeze-excitation
with simpler channel attention and uses ReLU6 for quantization friendliness.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn

from flashedge.registry import MODELS


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution (MBConv) block.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Depthwise convolution kernel size.
        stride: Depthwise convolution stride.
        expand_ratio: Expansion ratio for the bottleneck.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        mid_channels = int(in_channels * expand_ratio)

        layers: list = []

        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU6(inplace=True),
            ])

        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


@MODELS.register("efficientnet_lite")
class EfficientNetLite(nn.Module):
    """EfficientNet-Lite — a quantization-friendly variant of EfficientNet.

    Uses ReLU6 instead of Swish and removes squeeze-excitation for better
    compatibility with INT8 quantization on mobile/edge hardware.

    Args:
        num_classes: Number of output classes.
        width_mult: Width multiplier.
        depth_mult: Depth multiplier.
    """

    # (expand_ratio, channels, num_blocks, stride, kernel_size)
    DEFAULT_CONFIG: List[Tuple[float, int, int, int, int]] = [
        (1, 16, 1, 1, 3),
        (6, 24, 2, 2, 3),
        (6, 40, 2, 2, 5),
        (6, 80, 3, 2, 3),
        (6, 112, 3, 1, 5),
        (6, 192, 4, 2, 5),
        (6, 320, 1, 1, 3),
    ]

    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
    ) -> None:
        super().__init__()

        def c(channels: int) -> int:
            return max(int(channels * width_mult), 8)

        def d(num_blocks: int) -> int:
            return max(int(math.ceil(num_blocks * depth_mult)), 1)

        stem_channels = c(32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU6(inplace=True),
        )

        blocks: list = []
        in_channels = stem_channels
        for expand_ratio, channels, num_blocks, stride, kernel_size in self.DEFAULT_CONFIG:
            out_channels = c(channels)
            num = d(num_blocks)

            for i in range(num):
                s = stride if i == 0 else 1
                blocks.append(MBConvBlock(in_channels, out_channels, kernel_size, s, expand_ratio))
                in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)

        head_channels = c(1280)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(head_channels, num_classes),
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
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
